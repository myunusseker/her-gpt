import os
import re
import json

from src.gpts.gpt_client import GPTClient
from src.responses.action_response import ActionResponse


class ActionGeminiClient(GPTClient):
    """
    Gemini variant of the action planner. Builds a few-shot prompt with images and asks for a 2D action (dx, dy).
    Images (observation/goal and in examples) are expected as file paths. They are loaded via upload_image for Gemini.
    """

    def __init__(self, model_name="gemini-2.5-flash", task_description=None, data_root=os.path.join("data", "test", "cube_data")):
        with open("prompts/action_gpt_system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().replace("<TASK_DESCRIPTION_HERE>", task_description)
        super().__init__(model_name=model_name, system_prompt=system_prompt, gpt_type="gemini")
        self.data_root = data_root

    def _build_contents(self, error_category, observation, goal, examples=None):
        contents = []

        contents.append(f"We are currently encountering this error in the task: {error_category}.")
        contents.append("Your objective is to propose a single action vector (dx dy) to correct it.")

        if examples:
            contents.append("#### Here are some examples of this error type that you can utilize:")
            for i, (ex_obs, ex_goal, ex_action) in enumerate(examples, 1):
                contents.append(f"## Example {i}:")
                contents.append("Observation:")
                contents.append(ex_obs)
                contents.append("Goal:")
                contents.append(ex_goal)
                contents.append(f"Action: {ex_action}")

        contents.append("####\nFor the current observation and goal, respond ONLY with a JSON object of the form:\n{\"parameters\": [dx, dy], \"reasoning\": \"...\"}.")
        contents.append("Current Observation:")
        contents.append(observation)
        contents.append("Current Goal:")
        contents.append(goal)
        contents.append("Action:")

        return contents

    def propose_action(self, error_category, observation, goal, examples=None):
        contents = self._build_contents(error_category, observation, goal, examples=examples)
        text = self.chat(contents)

        # Try structured JSON first
        parsed = None
        cleaned = text.strip()
        # Strip code fences if present
        if cleaned.startswith("```json") or cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            data = json.loads(cleaned)
            params = data.get("parameters", [])
            reasoning = data.get("reasoning", "")
            # Coerce to two floats
            try:
                params = [float(params[0]), float(params[1])]
            except Exception:
                # Fallback to numeric scrape inside JSON text fields
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", json.dumps(data))
                if len(nums) >= 2:
                    params = [float(nums[0]), float(nums[1])]
                elif len(nums) == 1:
                    params = [float(nums[0]), 0.0]
                else:
                    params = [0.0, 0.0]
            return ActionResponse(parameters=params, reasoning=reasoning), None
        except Exception:
            pass

        # Fallback: parse first two floats from free text
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) >= 2:
            params = [float(nums[0]), float(nums[1])]
        elif len(nums) == 1:
            params = [float(nums[0]), 0.0]
        else:
            params = [0.0, 0.0]
        return ActionResponse(parameters=params, reasoning=text), None


if __name__ == "__main__":
    client = ActionGeminiClient(model_name="gemini-2.5-flash", task_description="The task is to place a cube on the circular target position.")
    obs_id = client.upload_image("data/test/cube_data/side_views/side_view_330.png")
    goal_id = client.upload_image("data/cube_goal.png")
    response, _ = client.propose_action(
        error_category="cube is too far",
        observation=obs_id,
        goal=goal_id,
    )
    print("Proposed action:", response.parameters)
    print("Reasoning:", response.reasoning)
