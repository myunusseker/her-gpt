from src.gpts.gpt_client import GPTClient


class ErrorOpenAIClient(GPTClient):
    """Error GPT client that classifies environmental observations into error categories."""
    
    def __init__(self, model_name="gpt-4o", task_description=None, use_history=False):
        with open("prompts/error_gpt_system.txt", "r") as file:
            system_prompt = file.read().replace("<TASK_DESCRIPTION_HERE>", task_description)
        print(f"Using system prompt: {system_prompt}")
        self.previous_response_id = None
        self.use_history = use_history
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

    def classify_error(self, observation, goal, error_categories):
        messages = self.generate_error_prompt(observation, goal, error_categories)
        
        response_text, response_id = self.chat(messages, self.previous_response_id if self.use_history else None)
        self.previous_response_id = response_id
        
        error_category = response_text.strip().lower()

        return error_category
    
if __name__ == "__main__":
    # Example usage
    error_gpt = ErrorOpenAIClient(
        model_name="gpt-5",
        task_description="The task is to place a cube on the circular target position."
    )

    observation_id = error_gpt.upload_image("data/test/cube_data/side_views/side_view_330.png")
    goal_id = error_gpt.upload_image("data/cube_goal.png")
    error_categories=[
        "cube is too far",
        "cube is to the left of the target",
        "cube is to the right of the target",
        "cube is in front of the target",
        "cube is behind the target",
    ]
    
    for i in range(10):
        error_category = error_gpt.classify_error(observation_id, goal_id, error_categories)
        print(f"\n#################\nError Category: {error_category}")