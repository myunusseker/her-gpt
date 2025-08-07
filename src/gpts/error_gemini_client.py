import base64
from google import genai
from google.genai import types
from PIL import Image

from src.gpts.gpt_client import GPTClient


class ErrorGeminiClient(GPTClient):
    """Error GPT client that classifies environmental observations into error categories using Gemini API."""

    def __init__(self, model_name="gemini-2.5-flash", task_description=None):
        with open("prompts/error_gpt_system.txt", "r") as file:
            system_prompt = file.read().replace("<TASK_DESCRIPTION_HERE>", task_description)
        print(f"Using system prompt: {system_prompt}")
        super().__init__(model_name, system_prompt, gpt_type="gemini")

    def generate_error_prompt(self, observation_id, goal_id, error_categories):
        categories_str = '\n'.join(f"- {cat}" for cat in error_categories)
        # Return a flat list of prompt parts: text strings and image bytes
        messages = [
            "\n\nObservation:",
            observation_id,
            f"\n\nGoal:",
            goal_id,
            f"\n\nError categories:\n{categories_str}\nYour answer:"
        ]
        return messages

    def classify_error(self, observation, goal, error_categories):
        messages = self.generate_error_prompt(observation, goal, error_categories)
        response_text = self.chat(messages)
        error_category = response_text.strip().lower()
        return error_category

if __name__ == "__main__":
    error_gpt = ErrorGeminiClient(model_name="gemini-2.5-flash", task_description="The task is to place a cube on the circular target position.")
    observation_id = error_gpt.upload_image("data/test/cube_data/side_views/side_view_330.png")
    goal_id = error_gpt.upload_image("data/test/cube_data/cube_goal.png")

    error_categories = [
        "cube is too far",
        "cube is to the left of the target",
        "cube is to the right of the target",
        "cube is at upper side of the target",
        "cube is at below side of the target",
    ]

    error_category = error_gpt.classify_error(observation_id, goal_id, error_categories)
    print(f"Error Category: {error_category}")