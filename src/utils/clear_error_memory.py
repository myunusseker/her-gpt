# Clear error memory for the given task
import os
def clear_error_memory(task_name="cube_sliding"):
    """
    Clear the error memory for the specified task by deleting all files in the observations, actions, and goals under error data folders.
    """
    base_path = f"data/{task_name}/errors"
    
    if not os.path.exists(base_path):
        print(f"No error data found for task: {task_name}")
        return
    
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        print(f"Clearing memory for category: {category}")
        if os.path.isdir(category_path):
            observation_folder = os.path.join(category_path, "observations")
            goal_folder = os.path.join(category_path, "goals")
            action_folder = os.path.join(category_path, "actions")
            
            # Clear observations
            if os.path.exists(observation_folder):
                for file in os.listdir(observation_folder):
                    os.remove(os.path.join(observation_folder, file))
            
            # Clear goals
            if os.path.exists(goal_folder):
                for file in os.listdir(goal_folder):
                    os.remove(os.path.join(goal_folder, file))
            
            # Clear actions
            if os.path.exists(action_folder):
                for file in os.listdir(action_folder):
                    os.remove(os.path.join(action_folder, file))
    
    print(f"Cleared error memory for task: {task_name}")

if __name__ == "__main__":
    clear_error_memory("cube_sliding")