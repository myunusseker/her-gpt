import os
from PIL import Image
import numpy as np
from src.environments.cube_environment import CubeEnvironment
from src.gpts.error_gemini_client import ErrorGeminiClient
from src.gpts.error_openai_client import ErrorOpenAIClient

class CubeSliding:
    # Initialize the CubeSliding task with necessary parameters
    def __init__(
            self, 
            gui=True, 
            task_description=None, 
            error_gpt_type="gemini", 
            error_model_name="gemini-2.5-flash", 
            error_use_history=False,
            error_categories=None,
            action_gpt_type="gemini",
            action_model_name="gemini-2.5-flash",
            action_use_history=False,
        ):
        self.gui = gui
        self.task_description = task_description
        self.error_gpt_type = error_gpt_type
        self.error_model_name = error_model_name
        self.error_use_history = error_use_history
        self.error_categories = error_categories
        self.action_gpt_type = action_gpt_type
        self.action_model_name = action_model_name
        self.action_use_history = action_use_history

        self.env = CubeEnvironment(gui=gui)
        
        if self.error_gpt_type == "gemini":
            self.error_gpt = ErrorGeminiClient(
                model_name=self.error_model_name, 
                task_description=self.task_description, 
                use_history=self.error_use_history
            )
        else:
            self.error_gpt = ErrorOpenAIClient(
                model_name=self.error_model_name, 
                task_description=self.task_description,
                use_history=self.error_use_history
            )
        
        #TODO: initialize action_gpt
        # if self.action_gpt_type == "gemini":
        #     self.action_gpt = ActionGeminiClient(
        #         model_name=self.action_model_name,
        #         task_description=self.task_description,
        #         use_history=self.action_use_history
        #     )
        # else:
        #     self.action_gpt = ActionOpenAIClient(
        #         model_name=self.action_model_name,
        #         task_description=self.task_description,
        #         use_history=self.action_use_history
        #     )

        self.initialize_data_folders()

    def initialize_data_folders(self):
        self.data_folders = {
            category: f"data/cube_sliding/errors/{category.replace(' ', '_')}" for category in self.error_categories
        }
        self.observation_folders = {
            category: f"{folder}/observations" for category, folder in self.data_folders.items()
        }
        self.goal_folders = {
            category: f"{folder}/goals" for category, folder in self.data_folders.items()
        }
        self.action_folders = {
            category: f"{folder}/actions" for category, folder in self.data_folders.items()
        }

        for folder in self.data_folders.values():
            os.makedirs(folder, exist_ok=True)
        for folder in self.observation_folders.values():
            os.makedirs(folder, exist_ok=True)
        for folder in self.goal_folders.values():
            os.makedirs(folder, exist_ok=True)
        for folder in self.action_folders.values():
            os.makedirs(folder, exist_ok=True)

    def run_task(self):
        self.env.reset()
        error_category = self.error_categories[0]
        action = np.array([0, 0, 0])
        # In a loop, perform the following steps:
        for i in range(10):
            print(f"Iteration {i}: Applying action {list(action)}")
            self.env.apply_action(force_vector=action)
            self.env.render_views(
                topdown_path=f"data/cube_sliding/iterations/topdown_{i}.png", 
                side_path=f"data/cube_sliding/iterations/side_{i}.png"
            )
            
            #Check if the action was successful, if so, break the loop
            print("Task successful? y|N: ")
            if input().strip().lower() == 'y':
                break

            HER_obs = Image.open(f"data/cube_sliding/iterations/side_{max(i-1, 0)}.png") # This might be obs after reset...
            HER_goal = Image.open(f"data/cube_sliding/iterations/side_{i}.png")
            HER_action = action

            # Apply HER by saving the observation, goal, and action to the corresponding folders
            save_index = len(os.listdir(self.observation_folders[error_category]))
            HER_obs.save(f"{self.observation_folders[error_category]}/{save_index}.png") 
            HER_goal.save(f"{self.goal_folders[error_category]}/{save_index}.png")
            np.savetxt(f"{self.action_folders[error_category]}/{save_index}.txt", HER_action)

            current_obs = self.error_gpt.upload_image(f"data/cube_sliding/iterations/side_{i}.png")
            current_goal = self.error_gpt.upload_image("data/cube_sliding/goal.png")
            
            error_category = self.error_gpt.classify_error(
                observation=current_obs,
                goal=current_goal,
                error_categories=self.error_categories,
            )

            print(f"Classified error category: {error_category}")
            if i == 0:
                action = np.array([40, 40, 0])
            else:
                action = np.array([np.random.uniform(-50, 50), np.random.uniform(-50, 50), 0])
            action_str = input(f"Suggested action: {list(action)}. Enter new action as x y or press Enter to accept: ")
            if action_str.strip():
                try:
                    action = np.array([float(x) for x in action_str.strip().split()])  # Expecting input like "x y"
                    if len(action) != 2:
                        raise ValueError("Please enter exactly two values for x and y.")
                    action = np.append(action, 0)  # Append 0 for the z component
                except ValueError as ve:
                    print(f"Invalid input: {ve}. Using suggested action {list(action)}.")
            #TODO: Retrieve observation, goal, action samples from the data folder according to the error category
            current_obs = self.error_gpt.upload_image(f"data/cube_sliding/iterations/side_{i}.png")
            current_goal = self.error_gpt.upload_image("data/cube_sliding/goal.png")
            examples = self._retrieve_examples(error_category)
            #TODO: Send examples to action_gpt to generate an action
            action, response_id = self.action_gpt.propose_action(
                error_category=error_category,
                observation=current_obs,
                goal=current_goal,
                examples=examples
            )


if __name__ == "__main__":
    # Example usage of CubeSliding task
    cube_sliding_task = CubeSliding(
        gui=True, 
        task_description="The task is to slide a cube to the target position.", 
        error_gpt_type="gemini", 
        error_model_name="gemini-2.5-flash", 
        error_use_history=True,
        error_categories=[
            "cube is too far",
            "cube is to the left of the target",
            "cube is to the right of the target",
            "cube is in front of the target",
            "cube is behind the target",
        ]
    )

    cube_sliding_task.run_task()