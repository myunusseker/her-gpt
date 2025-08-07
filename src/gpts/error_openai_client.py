from src.gpts.gpt_client import GPTClient


class ErrorOpenAIClient(GPTClient):
    """Error GPT client that classifies environmental observations into error categories."""
    
    def __init__(self, model_name="gpt-4o", task_description=None):
        with open("prompts/error_gpt_system.txt", "r") as file:
            system_prompt = file.read().replace("<TASK_DESCRIPTION_HERE>", task_description)
        print(f"Using system prompt: {system_prompt}")
        super().__init__(model_name, system_prompt, gpt_type="openai")

    def generate_error_prompt(self, observation_id, goal_id, error_categories):
        categories_str = '\n'.join(f"- {cat}" for cat in error_categories)
        content = [
            {"type": "input_text", "text": "\n\nObservation:"},
            {"type": "input_image", "file_id": observation_id},
            {"type": "input_text", "text": f"\n\nGoal:"},
            {"type": "input_image", "file_id": goal_id},
            {"type": "input_text", "text": f"\n\nError categories:\n{categories_str}\nYour answer:"}
        ]
        messages = [{"role": "user", "content": content}]
        return messages

    def classify_error(self, observation, goal, error_categories, previous_response_id=None):
        messages = self.generate_error_prompt(observation, goal, error_categories)
        
        response_text, response_id = self.chat(messages, previous_response_id)
        
        error_category = response_text.strip().lower()

        return error_category, response_id
    
if __name__ == "__main__":
    # Example usage
    error_gpt = ErrorOpenAIClient(model_name="gpt-4.1",task_description="The task is to place a cube on the circular target position.")
    observation_id = error_gpt.upload_image("data/test/cube_data/side_views/side_view_330.png")
    goal_id = error_gpt.upload_image("data/test/cube_data/cube_goal.png")

    error_categories = [
        "cube is too far",
        "cube is to the left of the target",
        "cube is to the right of the target",
        "cube is at the upper side of the target",
        "cube is at the below side of the target",
    ]
    
    for i in range(10):
        error_category, response_id = error_gpt.classify_error(observation_id, goal_id, error_categories)
        print(f"Error Category: {error_category}")
    # gpt_client = GPTClient(model_name="gpt-4o", system_prompt="You will explain your previous reasonings.")
    # user_text = input("Enter your message: ")
    # response_text, response_id = gpt_client.chat([{ "role": "user", "content": user_text }], previous_response_id=response_id)
    # print(f"Response Text: {response_text}, Response ID: {response_id}")