from goalchain import Field, ValidationError, Goal, Action, GoalChain 
import os, random
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv() 
# or set directly 
# os.environ["OPENAI_API_KEY"] = "sk-ABC..." # SPECIFY OPENAI API KEY

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

product_order_goal = ProductOrderGoal(
    label="product_order",
    goal="to obtain information on an order to be made",
    opener="I see you are trying to order a product, how can I help you?",
    out_of_scope="Ask the user to contact sales team at sales@acme.com"
)

class OrderCancelGoal(Goal):
    reason = Field("reason for order cancellation (optional)", format_hint="a string")
    
order_cancel_goal = OrderCancelGoal(
    label="cancel_current_order",
    goal="to obtain the reason for the cancellation",
    opener="I see you are trying to cancel the current order, how can I help you?",
    out_of_scope="Ask the user to contact the support team at support@acme.com",
    confirm=False
)

# Define the processing function
def process_order(data):
    # Simulate processing the order
    order_number = "ORD123456"
    data['order_number'] = order_number
    return data

# Create the Action with rephrase enabled
process_order_action = ~Action(
    function=process_order,
    response_template="Your order has been processed successfully! Your order number is {{ order_number }}.",
    rephrase=True,
)

# Define the processing function
def cancel_order(data):
    # Simulate processing the order
    order_number = "ORD123456"
    data['order_number'] = order_number
    return data

# Create the Action with rephrase enabled
cancel_order_action = ~Action(
    function=cancel_order,
    response_template="Your order number {{ order_number }} has been cancelled successfully. I understand the reason you provided: {{ reason }}",
    rephrase=True,
)

# Define a condition function
def is_high_quantity(data):
    return data['quantity'] >= 50

class HighValueOrderGoal(Goal):
    verification_code = Field("verification code", format_hint="a 6-digit code")

    def on_start(self):
        # Send verification code
        verification_code = send_verification_code(self.data['customer_email'])
        self.expected_code = verification_code

    def on_complete(self, data):
        if data['verification_code'] == self.expected_code:
            # Verification successful, proceed to processing
            self.next_action = process_high_value_order_action
            return data
        else:
            # Verification failed, ask again
            response = self.simulate_response("Incorrect verification code. Please try again.", rephrase=True)
            return response  # Return the response directly

high_value_order_goal = HighValueOrderGoal(
    label="high_value_order",
    goal="to verify high-value orders",
    opener="Since you're ordering a large quantity, we need to verify your order by sending you a verification code over email.",
    out_of_scope="Please contact support for further assistance."
)

# Chain the goals and actions
product_order_goal >> process_order_action
product_order_goal >> order_cancel_goal / "to cancel the current order"
order_cancel_goal >> cancel_order_action
order_cancel_goal >> product_order_goal / "to continue with the order anyway"

# Process high-value order action
def process_high_value_order(data):
    # Simulate processing the high-value order
    order_number = "ORD789012"
    data['order_number'] = order_number
    return data

process_high_value_order_action = ~Action(
    function=process_high_value_order,
    response_template="Your high-value order has been verified and processed successfully! Your order number is {{ order_number }}.",
    rephrase=True,
)

def send_verification_code(email):
    verification_code = str(random.randint(100000, 999999))
    # Simulate sending the code via email
    print(f"[[[ Verification code sent to {email}: {verification_code} ]]]")
    return verification_code

process_order_action >> (is_high_quantity, high_value_order_goal)
high_value_order_goal >> process_high_value_order_action
high_value_order_goal >> (order_cancel_goal / "to cancel the current order")

goal_chain = GoalChain(product_order_goal)

print(goal_chain.get_response()["content"])
while True:
    user_input = input("You: ")
    response = goal_chain.get_response(user_input)
    print("Assistant: " + response["content"])
    if response["type"] == "end":
        # Print a final message
        print("Assistant: " + goal_chain.simulate_response("Thank you for choosing our service. Have a great day!", rephrase=True)["content"])
        break