import os
from dotenv import load_dotenv
from openai import OpenAI

from src.responses.action_response import ActionResponse


class GPTClient:
    """Wrapper around the OpenAI client for vision-based prompts."""

    def __init__(self, model_name, system_prompt = None):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.system_prompt = system_prompt

    def upload_image(self, path):
        with open(path, "rb") as f:
            result = self.client.files.create(file=f, purpose="vision")
        return result.id

    def chat(self, messages, previous_response_id = None, text_format = None):
        if text_format is None:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
            )
            return response.output_text, response.id
        else:
            response = self.client.responses.parse(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
                text_format=text_format,
            )
            return response.output_parsed, response.id

if __name__ == "__main__":
    # Example usage
    client = GPTClient(model_name="gpt-4.1", system_prompt="You are a helpful assistant.")
    for i in range(10):
        input_text = input("Enter your message: ")
    
        response_text, response_id = client.chat([{
            "role": "user", 
            "content": [
                {"type": "input_text", "text": input_text}
            ]}], 
            previous_response_id=response_id if i > 0 else None
        )
        print(f"Response: {response_text}")
