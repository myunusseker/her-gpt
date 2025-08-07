import os
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from src.responses.action_response import ActionResponse


class GPTClient:
    """Wrapper around the OpenAI client for vision-based prompts."""

    def __init__(self, model_name, system_prompt = None, gpt_type="openai"):
        load_dotenv()
        if gpt_type == "gemini":
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.gpt_type = gpt_type

    def upload_image(self, path):
        # For Gemini, we can directly use the image path
        if self.gpt_type == "gemini":
            return Image.open(path)
        
        # For OpenAI, we need to upload the image and get a file ID
        with open(path, "rb") as f:
            result = self.client.files.create(file=f, purpose="vision")
        return result.id

    def chat(self, messages, previous_response_id = None, text_format = None, full_response = False):
        # For Gemini, we use the generate_content method
        if self.gpt_type == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name, 
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt
                ),
                contents=messages
            )
            return response.text
        
        # For OpenAI, we use the chat method
        if text_format is None:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
            )
            return response.output_text if not full_response else response, response.id
        else:
            response = self.client.responses.parse(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
                text_format=text_format,
            )
            return response.output_parsed if not full_response else response, response.id

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
