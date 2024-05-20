# GoalChain
GoalChain is a simple but effective framework for enabling goal-orientated conversation flows for human-LLM and LLM-LLM interaction.

## Installation

```console
pip install goalchain
```

## Getting started

Let's import the `Field`, `ValidationError`, `Goal` and `GoalChain` classes, which are the basis for the conversation flow.
```py
from goalchain import Field, ValidationError, Goal, GoalChain 
```

In this example we will create an AI assistant whose goal is to collect information from a customer about their desired product order. We define the information to be collected using `Field` objects within the `ProductOrderGoal`, which is a child of `Goal`: 
* the product name,
* the customer's email, and 
* quantity

We also define a validator for the quantity (after type casting to an int). `ValidationError` is used to pass error messages back to conversation. These messages should be human-readable.

`format_hint` is a natural language type hint for the LLM's JSON mode output.

```py
def quantity_validator(value):
    try:
        value = int(value)
    except (ValueError, TypeError):
        raise ValidationError("Quantity must be a valid number")
    if value <= 0:
        raise ValidationError("Quantity cannot be less than one")
    if value > 100:
        raise ValidationError("Quantity cannot be greater than 100")
    return value

class ProductOrderGoal(Goal):
    product_name = Field("product to be ordered", format_hint="a string")
    customer_email = Field("customer email", format_hint="a string")
    quantity = Field("quantity of product", format_hint="an integer", validator=quantity_validator)
```

In case the customer changes their mind, let's create another `Goal` child class called `OrderCancelGoal`.
We will request an optional reason for the customer's cancellation of the ongoing order. Through specifying that the field is "(optional)" in the description, the LLM will know it isn't necessary to achieve the goal. 

```py
class OrderCancelGoal(Goal):
    reason = Field("reason for order cancellation (optional)", format_hint="a string")
```

Note that the field object names, such as `product_name` are passed directly to the LLM prompt, and so they are part of the prompt-engineering task, as is every other string. 

Essentially the classes we defined are like forms to be filled out by the customer, but they lack instructions. Let's add those by instantiating the classes as objects. 

```py
product_order_goal = ProductOrderGoal(
    label="product_order",
    goal="to obtain information on an order to be made",
    opener="I see you are trying to order a product, how can I help you?",
    out_of_scope="Ask the user to contact sales team at sales@acme.com"
)

order_cancel_goal = OrderCancelGoal(
    label="cancel_current_order",
    goal="to obtain the reason for the cancellation",
    opener="I see you are trying to cancel the current order, how can I help you?",
    out_of_scope="Ask the user to contact the support team at support@acme.com",
    confirm=False
)
```

We define
* an internal label to be used (also part of our prompt-engineering task),
* the goal, expressed as a "to ..." statement,
* a default `opener` - something the AI assistant will use given no prior input,
* and importantly, instructions for the AI assistant as to what they should do in case of an out of scope user query

The `confirm` flag determines whether the AI assistant will ask for confirmation once it has all of the required information defined using the `Field` objects. It is `True` by default. We don't need a confirmation for the order cancellation goal, since it is in itself already a kind of confirmation.

Next we need to connect the goals together.

```py
product_order_goal.connect(goal=order_cancel_goal, 
                           user_goal="to cancel the current order", 
                           hand_over=True, 
                           keep_messages=True)

```

The `user_goal` is another "to ..." statement. Without `hand_over=True` the AI agent would reply with the canned `opener`. Setting it to `True` ensures the conversation flows smoothly. Sometimes you may want a canned response, other times not. 

`keep_messages=True` means the `order_cancel_goal` will receive the full history of the conversation with `product_order_goal`, otherwise it will be wiped. Again, sometimes a wipe of the conversation history may be desired, such as when simulating different AI personalities.

Let's also consider the possibility of a really undecisive customer. We should also give them the option to "cancel the cancellation". 

```py
order_cancel_goal.connect(goal=product_order_goal, 
                          user_goal="to continue with the order anyway", 
                          hand_over=True, 
                          keep_messages=True)
```

At some point you may have wondered if you can make a goal without any `Field` objects. You can! Such a goal is a routing goal defined only be the connections it has. This is useful for example in a voice-mail menu system. 

You may also be curious whether you can connect a goal to itself. You can! This is useful for example when using `confirm=False` with the `Goal`-inheriting object, where you require sequential user input of some variety. 

You can also chain connects, e.g. `goal.connect(...).connect(...).connect(...)`

Finally, let's use `GoalChain` to set the initial goal and test our AI sales assistant!

```py
goal_chain = GoalChain(product_order_goal)
```

Note that each goal can use a separate LLM API as enabled by [LiteLLM](https://github.com/BerriAI/litellm), and if you have the required environment variables set, you can use any model from the supported [model providers](https://docs.litellm.ai/docs/providers).

The default model is `"gpt-4-1106-preview"`, that is:

```py
product_order_goal = ProductOrderGoal(...
    model="gpt-4-1106-preview", 
    json_model="gpt-4-1106-preview"
)
```

So we also need to set the `OPENAI_API_KEY` environmental variable.

```py
import os
os.environ["OPENAI_API_KEY"] = "sk-ABC..."
```

Usually it is the user who prompts the AI agent first, but if this is not the case, we call `get_response` without any arguments, or use `None`:

```py
goal_chain.get_response()
```

```txt
{'type': 'message',
 'content': 'Great choice! Could you please provide me with your email address to proceed with the order?',
 'goal': <__main__.ProductOrderGoal at 0x7f8c8b687110>}
```

GoalChain returns a `dict` containing the type of response (either `message` or `data`), the content of the response (right now just our canned response) and the current `Goal`-inheriting object.

Let's query our AI assistant with a potential purchase.

```py
goal_chain.get_response("Hi, I'd like to buy a vacuum cleaner")
```

```txt
{'type': 'message',
 'content': 'Great! Could you please provide your email address so we can send the confirmation of your order?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

The AI assistant is working towards achieving its current goal, and gathering the required information for an order. 

```py
goal_chain.get_response("Sure, it is john@smith.com")
```

```txt
{'type': 'message',
 'content': 'Thank you, John. Which model of vacuum cleaner would you like to order?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

```py
goal_chain.get_response("The 2000XL model")
```

```txt
{'type': 'message',
 'content': 'How many of the 2000XL model would you like to order?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
 
```

Let's test whether our AI assistant can handle a cancellation of the current order.

```py
goal_chain.get_response("Actually I changed my mind, cancel this order")
```

```txt
{'type': 'message',
 'content': 'Of course, I can assist with that. Could you please tell me the reason for the cancellation?',
 'goal': <__main__.OrderCancelGoal at 0x7ff0fb275650>}
 
```

It worked. Note that the returned goal is now of type `OrderCancelGoal`. We've switched goals. Let's also test whether we can switch back. 

```py
goal_chain.get_response("Actually, yeah, I would like to buy the vacuum cleaner")
```

```txt
{'type': 'message',
 'content': 'Understood. How many of the 2000XL model would you like to order?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

We're back to the `ProductOrderGoal`.

```py
goal_chain.get_response("1 please")
```

```txt
{'type': 'message',
 'content': 'To confirm, you would like to order one 2000XL vacuum cleaner and the order will be sent to john@smith.com, is that correct?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

The AI assistant confirms our order. If we didn't like this behaviour we would use `confirm=False`.

Let's see how the assistant responds to an out of scope query.

```py
goal_chain.get_response("Is it a good vacuum cleaner? What do you think?")
```

```txt
{'type': 'message',
 'content': "For product reviews and additional information, I recommend contacting our sales team at sales@acme.com. They can help with your inquiries. Meanwhile, can you please confirm if you'd like to proceed with the order for one 2000XL vacuum cleaner to john@smith.com?",
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

The AI assistant redirects us to the sales team inbox as defined earlier, and re-iterates the confirmation.

But let's throw a curve-ball...
 
```py
goal_chain.get_response("Ok, I'd actually like to make that an order of 500")
```

```txt
{'type': 'message',
 'content': "Just to clarify, you'd like to order 500 units of the 2000XL vacuum cleaner, with the order confirmation sent to john@smith.com. Is that correct?",
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```
 
```py
goal_chain.get_response("Yes")
```

```txt
{'type': 'message',
 'content': 'I’m sorry, but I need to inform you that the quantity cannot be greater than 100 for an order. If you would like to proceed with an order within this limit, please let me know.',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

The validator we use has given enough information to the AI assistant to justify why it cannot process this quantity via the `ValidationError` message.

Let's complete the order.

```py
goal_chain.get_response("Alright, I'll guess I'll just go with 1")
```

```txt
{'type': 'message',
 'content': 'To confirm, you would like to order one 2000XL vacuum cleaner and the order will be sent to john@smith.com, is that correct?',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
``` 

```py
goal_chain.get_response("That's right")
```

```txt
{'type': 'data',
 'content': {'customer_email': 'john@smith.com',
  'product_name': '2000XL',
  'quantity': 1},
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
``` 

The content returned is a dictionary parsed from the output of the LLM's JSON mode. The keys are our field instance names. We can now use the data to perform some kind of action, such as processing the order of our hypothetical 2000XL vacuum cleaner. 

Note that in reality, if you were building such a system, you would need to make a dedicated product-lookup goal as not to allow arbitrary or meaningless product names. 

Let's send our confirmation the order has been processed via `simulate_response`. We will also use  `rephrase = True` to rephrase the output, which will appear more natural in case the customer frequently interacts with the goal. 
 
```py
goal_chain.simulate_response(f"Thank you for ordering from Acme. Your order will be dispatched in the next 1-3 business days.", rephrase = True)
```

```txt
{'type': 'message',
 'content': 'We appreciate your purchase with Acme! Rest assured, your order will be on its way within the next 1 to 3 business days.',
 'goal': <__main__.ProductOrderGoal at 0x7ff0fb283090>}
```

At this point we may end the session or connect back to a menu or routing goal for further input. 

If you would like to customise or contribute to GoalChain, or report any issues, visit the [GitHub page](https://github.com/adlumal/GoalChain).  

