import os

from src.gpts.gpt_client import GPTClient
from src.responses.action_response import ActionResponse


class ActionOpenAIClient(GPTClient):
    """
    Agent that proposes an action given an error category, current observation image, and goal image.
    Uses GPTClient (OpenAI Responses API) and builds few-shot vision prompts from provided triplet examples.
    """

    def __init__(self, model_name="gpt-5", task_description=None, data_root=os.path.join("data", "test", "cube_data")):
        with open("prompts/action_gpt_system.txt", "r") as file:
            system_prompt = file.read().replace("<TASK_DESCRIPTION_HERE>", task_description)
        super().__init__(model_name=model_name, system_prompt=system_prompt, gpt_type="openai")
        self.data_root = data_root

    
    def _build_messages(self, error_category, observation, goal, examples=None):
        content = []
        
        content.extend([
            {"type": "input_text", "text": f"We are currently encountering this error in the task: {error_category}."},
            {"type": "input_text", "text": "Your objective is to propose a single action vector (dx dy) to correct it."},
        ])

        if examples:
            content.append({"type": "input_text", "text": "#### Here are some examples of this error type that you can utilize:"})
            for i, (ex_obs, ex_goal, ex_action) in enumerate(examples, 1):
                content.append({"type": "input_text", "text": f"## Example {i}:"})
                content.append({"type": "input_text", "text": "Observation:"})
                content.append({"type": "input_image", "file_id": ex_obs})
                content.append({"type": "input_text", "text": "Goal:"})
                content.append({"type": "input_image", "file_id": ex_goal})
                content.append({"type": "input_text", "text": f"Action: {ex_action}"})

        content.append({"type": "input_text", "text": "####\n For the current observation and goal, provide an ActionResponse. parameters must be exactly two floats: dx dy."})
        content.append({"type": "input_text", "text": "Current Observation:"})
        content.append({"type": "input_image", "file_id": observation})
        content.append({"type": "input_text", "text": "Current Goal:"})
        content.append({"type": "input_image", "file_id": goal})
        content.append({"type": "input_text", "text": "Action:"})

        messages = [
            {"role": "user", "content": content},
        ]
        return messages

    def propose_action(self, error_category, observation, goal, examples=None):
        messages = self._build_messages(error_category, observation, goal, examples=examples)
        response, response_id = self.chat(messages, text_format=ActionResponse)
        return response, response_id

if __name__ == "__main__":
    client = ActionOpenAIClient(        
        model_name="gpt-5",
        task_description="The task is to place a cube on the circular target position."
    )
    # Upload images first and pass file_ids as inputs
    obs_id = client.upload_image("data/test/cube_data/side_views/side_view_330.png")
    goal_id = client.upload_image("data/cube_goal.png")
    response, response_id = client.propose_action(
        error_category="cube is too far",
        observation=obs_id,
        goal=goal_id,
    )
    print("Proposed action:", response.parameters)
    print("Reasoning:", response.reasoning)
    print("Response ID:", response_id)
