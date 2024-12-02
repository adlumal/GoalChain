import os
import json
import inspect
from litellm import completion
from jinja2 import Environment, select_autoescape

# Define special constants
RESET = object()
CLEAR = object()

# Prompt class based on banks by Massimiliano Pippi
class Prompt:
    env = Environment(
        autoescape=select_autoescape(
            enabled_extensions=("html", "xml"),
            default_for_string=False,
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    def __init__(self, text, filters = {}):
        for filter_label, filter_function in filters.items():
            self.env.filters[filter_label] = filter_function
        self._template = self.env.from_string(text)

    def text(self, data):
        rendered: str = self._template.render(data)
        return rendered

class Field:
    def __init__(self, description, format_hint=None, validator=None):
        self.description = description
        self.format_hint = format_hint
        self.validator = validator
        
class Goal:
    def _format_flag(self, flag):
        return f"<{flag}>"
    
    def __init__(self,
                 label,
                 goal,
                 opener, 
                 out_of_scope=None, 
                 confirm=True, 
                 goal_prompt_template=None,
                 completed_prompt_template=None,
                 error_prompt_template=None,
                 validation_prompt_template=None,
                 rephrase_prompt_template=None,
                 model="gpt-4-1106-preview", 
                 json_model="gpt-4-1106-preview",
                 params = {}):
        self.label = label
        self.goal = goal
        self.opener = opener
        self.confirm = confirm
        self.out_of_scope = out_of_scope
        self.model = model
        self.json_model = json_model
        self.messages = []
        self.connected_goals = []
        self.completed_string = "completed"
        self.hand_over = True    
        self.params = params
        self.next_action = None 
        self.started = False
        
        self.goal_prompt = goal_prompt_template if goal_prompt_template else Prompt("""Your role is to continue the conversation below as the Assistant.
Goal: {{goal}}
{% if information_list %}
Information to be gathered: {{information_list|join(", ")}}
This is all of the information you are to gather, do not ask for anything else.
{% if confirmation %}
Once you have the information ask for a confirmation.
If you receive this confirmation reply only with:
{{ completed_string | format_flag }}
{% else %}
Once you have the information reply only with:
{{ completed_string | format_flag }}
{% endif %}
{% endif %}
{% if out_of_scope %}
{% for goal in connected_goals %}
If the user wants {{ goal.user_goal }} reply only with this:
{{ goal.goal.label | format_flag }}
{% endfor %}
For anything outside of the scope of the goal:
{{ out_of_scope }}
{% endif %}
Respond naturally, and don't repeat yourself.
Conversation so far:
{% for message in messages %}
{{ message.actor }}: {{ message.content }}
{% endfor %}
Assistant:""", filters={"format_flag": self._format_flag})
        self.completed_prompt = completed_prompt_template if completed_prompt_template else Prompt("""Given the conversation below output JSON which includes only the following keys:
{% for field in fields %}
{{ field.name }}: {{ field.description }} {% if field.format_hint %}({{field.format_hint}})
{% endif %}
{% endfor %}
If any keys are not provided in the conversation set their values to null.
Conversation:
{% for message in messages %}
{{ message.actor }}: {{ message.content }}
{% endfor %}""")
        self.error_prompt = error_prompt_template if error_prompt_template else Prompt("""I'm sorry, but I'm having trouble processing that request right now.""")
        self.validation_prompt = validation_prompt_template if validation_prompt_template else Prompt("""Your role is to continue the conversation below as the Assistant.
Unfortunately you had trouble processing the user's request because of the following problems:
{% for error in validation_error_messages %}
* {{ error }}
{% endfor %}
Continue the conversation naturally, and explain the problems.
Do not be creative. Do not make suggestions as to how to fix the problems.
Conversation so far:
{% for message in messages %}
{{ message.actor }}: {{ message.content }}
{% endfor %}
Assistant:""")
        self.rephrase_prompt = rephrase_prompt_template if rephrase_prompt_template else Prompt("""Your role is to continue the conversation below as the Assistant.
Normally you respond with: {{ response }}
{% if message_history %}
Goal: {{goal}}
But now you need to take into account the conversation so far and tailor your response accordingly.
Continue the conversation naturally. Do not be creative.
Conversation so far:
{% for message in message_history %}
{{ message.actor }}: {{ message.content }}
{% endfor %}
{% else %}
Simply rephrase your response as the Assistant.
{% endif %}
Assistant:""")
        
    # Overloading the '/' operator to create GoalConnection
    def __truediv__(self, other):
        if isinstance(other, str):
            # Create a GoalConnection with user_goal
            return GoalConnection(goal=self, user_goal=other)
        else:
            raise TypeError("Use 'goal / \"user goal description\"'")
        
    def __rshift__(self, other):
        if isinstance(other, GoalConnection):
            self.connect(
                goal=other.goal,
                user_goal=other.user_goal,
                hand_over=other.hand_over,
                keep_messages=other.keep_messages
            )
            return other.goal
        elif isinstance(other, Action):
            self.next_action = other
            return other
        else:
            raise TypeError("Use 'goal >> (goal2 / \"user goal description\" | RESET | CLEAR)' or 'goal >> Action(function)'")
        
    def get_fields(self):
        fields = inspect.getmembers(self)
        field_dict = {}
        for field in fields:
            if type(field[1]) == Field:
                field_dict[field[0]] = field[1]
        return field_dict

    def _get_goal_details(self):
        prompt_details = {
            "goal": self.goal,
            "confirmation": self.confirm,
            "messages": self.messages,
            "completed_string": self.completed_string,
            "out_of_scope": self.out_of_scope,
            "connected_goals": self.connected_goals,
        }
        
        fields = self.get_fields()
        information_list = []
        for label, field in fields.items():
            information_list.append(field.description)
        prompt_details["information_list"] = information_list
        return prompt_details
    
    def _get_completion_details(self):
        prompt_details = {
            "messages": self.messages,
        }
        
        fields = self.get_fields()
        field_list = []
        for label, field in fields.items():
            field_list.append(
                {
                    "name": label,
                    "description": field.description,
                    "format_hint": field.format_hint,
                }
            )
        prompt_details["fields"] = field_list
        return prompt_details
    
    def _inference(self, user_message, system_prompt="", json_mode=False):
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        model = self.json_model if json_mode else self.model
        response_format = {"type": "json_object"} if json_mode else None

        llm_response = completion(
            messages=llm_messages,
            model=model,
            response_format=response_format,
            **self.params
        )
        return llm_response["choices"][0]["message"]["content"]
    
    def on_complete(self, data):
        # Default behavior: proceed to the next action or return data
        return data
    
    def simulate_response(self, response, rephrase = False, message_history = []):
        if rephrase:
            rephrase_details = {
                "response": response,
                "message_history": message_history,
                "goal": self.goal,
            }
            rephrase_pre_prompt = self.rephrase_prompt.text(rephrase_details)
            response = self._inference(
                rephrase_pre_prompt
            )
        self.messages.append(
            {
                "actor": "Assistant",
                "content": response,
            }
        )
        return response
    
    def user_response(self, response):
        self.messages.append(
            {
                "actor": "User",
                "content": response,
            }
        )
        return response
    
    def get_response(self, user_input):
        if not self.messages and not user_input and not self.hand_over:
            return self.simulate_response(self.opener)
        elif not self.messages and not user_input and self.hand_over:
            return self.simulate_response(self.opener, rephrase=True)
        else:
            if user_input:
                user_input = self.user_response(user_input)
            response_text = self._inference(
                self.goal_prompt.text(self._get_goal_details())
            )
            
            # if HANDING OVER
            for connected_goal in self.connected_goals:
                if self._format_flag(connected_goal["goal"].label).lower() in response_text.lower():
                    if connected_goal["keep_messages"]:
                        hand_over_messages = self.messages
                    else:
                        hand_over_messages = []
                    return connected_goal["goal"].take_over(messages=hand_over_messages, hand_over=connected_goal["hand_over"])
            
            # if COMPLETED
            if self._format_flag(self.completed_string).lower() in response_text.lower():

                json_response_text = self._inference(
                    self.completed_prompt.text(self._get_completion_details()),
                    json_mode=True)
                
                try:
                    response_object = json.loads(json_response_text)
                    
                    validation_error_messages = []
                    fields = self.get_fields()
                    
                    for label, field in fields.items():
                        if label in response_object:
                            if field.validator:
                                try:
                                    response_object[label] = field.validator(response_object[label])
                                except ValidationError as e:
                                    validation_error_messages.append(str(e))
                                        
                    if not validation_error_messages:
                        return self.on_complete(response_object)
                    else:
                        validation_details = {
                            "validation_error_messages": validation_error_messages,
                            "messages": self.messages
                        }
                        validation_pre_prompt = self.validation_prompt.text(validation_details)
                        
                        validation_response_text = self._inference(
                            validation_pre_prompt
                        )
                        
                        return self.simulate_response(validation_response_text)
                    
                except json.JSONDecodeError:
                    error_response = self.error_prompt.text()
                    return self.simulate_response(error_response)

            else:
                return self.simulate_response(response_text)
    
    def take_over(self, messages=[], hand_over=True, data=None):
        if messages:
            self.messages = messages
        else:
            self.messages = []
        if hand_over:
            self.hand_over = True
        if data is not None:
            self.data = data
        else:
            self.data = {}
        if not self.started:
            if hasattr(self, 'on_start') and callable(self.on_start):
                self.on_start()
            self.started = True
        return self
    
    def connect(self, goal, user_goal, hand_over = True, keep_messages = True):
        self.connected_goals.append(
            {
                "goal": goal,
                "user_goal": user_goal,
                "hand_over": hand_over,
                "keep_messages": keep_messages,
            }
        )
        return goal
        
class ValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)

class GoalConnection:
    def __init__(self, goal, user_goal, hand_over=True, keep_messages=True):
        self.goal = goal
        self.user_goal = user_goal
        self.hand_over = hand_over
        self.keep_messages = keep_messages

    def __or__(self, other):
        if other is RESET:
            self.hand_over = False
            return self
        elif other is CLEAR:
            self.keep_messages = False
            return self
        else:
            raise TypeError("Use '| RESET' or '| CLEAR' to set options")

class Action:
    def __init__(self, function, response_template=None, rephrase=False, rephrase_prompt_template=None, conversation_end=False):
        self.function = function
        self.response_template = response_template
        self.rephrase = rephrase
        self.conversation_end = conversation_end
        self.next_goal = None
        self.conditions = []
        self.rephrase_prompt = rephrase_prompt_template if rephrase_prompt_template else Prompt("""
Your role is to continue the conversation below as the Assistant.
Please rephrase the following response to make it more natural and engaging, taking into account the conversation so far.
Continue the conversation naturally. Do not be creative. Do not include information, actions, suggestions, facts or details that are not in the response you are to rephrase.
Response:
{{ response }}
Conversation so far:
{% for message in message_history %}
{{ message.actor }}: {{ message.content }}
{% endfor %}
Assistant:
""")
        self.next_goal = None
        self.conditions = []  # List of tuples (condition_function, next_goal)

    # Overload the '~' operator to set conversation_end to True
    def __invert__(self):
        self.conversation_end = True
        return self

    # Overload '>>' to chain actions to goals or other actions
    def __rshift__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            condition_function, next_goal = other
            self.add_condition(condition_function, next_goal)
            return next_goal
        elif isinstance(other, Goal):
            self.next_goal = other
            return other
        elif isinstance(other, Action):
            self.next_action = other
            return other
        else:
            raise TypeError("Can only chain an Action to a Goal or another Action using '>>' operator")

    def add_condition(self, condition_function, next_goal):
        self.conditions.append((condition_function, next_goal))

    def execute(self, data, assistant):
        result = self.function(data)
        if self.response_template:
            # Generate response using the response_template
            response_text = self.generate_response(result)
            if self.rephrase:
                # Rephrase the response using the assistant's LLM
                response_text = self.rephrase_response(response_text, assistant)
            # After processing, check conditions
            for condition_function, next_goal in self.conditions:
                if condition_function(result):
                    self.next_goal = next_goal
                    break
            return response_text
        else:
            # No response template, but still check conditions
            for condition_function, next_goal in self.conditions:
                if condition_function(result):
                    self.next_goal = next_goal
                    break
            return result

    def generate_response(self, result):
        # Generate response using the response_template
        prompt = Prompt(self.response_template)
        response_text = prompt.text(result)
        return response_text

    def rephrase_response(self, response_text, assistant):
        rephrase_details = {
            'response': response_text,
            'goal': assistant.goal,
            'message_history': assistant.messages
        }
        rephrase_pre_prompt = self.rephrase_prompt.text(rephrase_details)
        # Use the assistant's inference method to get the rephrased response
        rephrased_response = assistant._inference(
            user_message=rephrase_pre_prompt,
            system_prompt="",
            json_mode=False
        )
        return rephrased_response

class GoalChain:
    def __init__(self, starting_goal):
        self.goal = starting_goal
        self.data = {}
        self.goal.take_over(data=self.data)

    def get_response(self, user_input=None):
        try:
            response = self.goal.get_response(user_input)
            if isinstance(response, str):
                return self._handle_message(response)
            elif isinstance(response, Goal):
                return self._handle_goal_transition(response)
            elif isinstance(response, dict):
                return self._handle_data_response(response)
            else:
                raise TypeError("Unexpected Goal response type")
        except Exception as e:
            return self.simulate_response("I'm sorry, something went wrong.")
    
    def _handle_message(self, response):
        return {"type": "message", "content": response, "goal": self.goal}

    def _handle_goal_transition(self, new_goal):
        # Check if transitioning to a different goal or re-entering the same goal
        if new_goal is not self.goal or new_goal.started:
            self.goal = new_goal
            self.goal.started = False  # Reset started flag
            self.goal.take_over(data=self.data)
        return self.get_response()
    
    def _handle_data_response(self, data):
        self.data.update(data)
        if hasattr(self.goal, 'next_action') and self.goal.next_action:
            action = self.goal.next_action
            action_response = action.execute(self.data, assistant=self.goal)
            # Check for next_goal first
            if hasattr(action, 'next_goal') and action.next_goal:
                self.goal = action.next_goal
                self.goal.take_over(data=self.data)
                return self.get_response()
            elif action.conversation_end:
                return {"type": "end", "content": action_response, "goal": self.goal}
            else:
                return {"type": "message", "content": action_response, "goal": self.goal}
        else:
            return {"type": "data", "content": self.data, "goal": self.goal}

    def simulate_response(self, user_input, rephrase = False):
        response = self.goal.simulate_response(user_input, rephrase = rephrase)
        return {"type": "message", "content": response, "goal": self.goal}