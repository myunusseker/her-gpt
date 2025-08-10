import json
import csv
import os
import base64
import numpy as np

from src.environments.cube_environment import CubeEnvironment

# Settings
n_samples = 100  # number of (observation, goal) pairs
dataset_name = "cube_pick_place_100_train"
out_jsonl = f"{dataset_name}.jsonl"
out_csv = f"{dataset_name}.csv"
save_debug_images = True  # only set True if you still want PNGs on disk for inspection
debug_dir = f"data/{dataset_name}_debug"
if save_debug_images:
    os.makedirs(debug_dir, exist_ok=True)

system_prompt_old = "You are a robotic agent that produces actions to move a cube based on observation and goal images."
system_prompt = (
    "You are a robotic agent that produces actions to move a cube based on observation and goal images.\n"
    "Input: You will be given an observation image and a goal image.\n"
    "Output format: action:[param1, param2], each formatted with exactly two decimals; no extra text.\n"
)

def grab_side_image_path(env, top_path_tmp, side_path_tmp):
    """Render both views; return side image path (caller encodes)."""
    env.render_views(topdown_path=top_path_tmp, side_path=side_path_tmp, use_static_side=True)
    # Optionally remove the top image if not debugging
    if not save_debug_images:
        try:
            os.remove(top_path_tmp)
        except OSError:
            pass
    return side_path_tmp

# Official image encode helper
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

env = CubeEnvironment(gui=False)

with open(out_jsonl, "w") as jf, open(out_csv, "w", newline="") as cf:
    csv_writer = csv.writer(cf)
    csv_writer.writerow([
        "system_prompt",
        "input_text_observation",
        "input_image_observation",
        "input_text_goal",
        "input_image_goal",
        "output_text",
    ])

    for i in range(n_samples):
        # Observation spawn
        obs_xy = np.round(np.random.uniform(0.0, 0.8, size=2), 2)
        env.reset(start_pos=[obs_xy[0], obs_xy[1], 0.02])
        tmp_top_obs = os.path.join(debug_dir if save_debug_images else ".", f"_obs_{i}_top.png")
        tmp_side_obs = os.path.join(debug_dir if save_debug_images else ".", f"_obs_{i}_side.png")
        obs_side_path = grab_side_image_path(env, tmp_top_obs, tmp_side_obs)

        # Goal spawn
        goal_xy = np.round(np.random.uniform(0.0, 0.8, size=2), 2)
        env.reset(start_pos=[goal_xy[0], goal_xy[1], 0.02])
        tmp_top_goal = os.path.join(debug_dir if save_debug_images else ".", f"_goal_{i}_top.png")
        tmp_side_goal = os.path.join(debug_dir if save_debug_images else ".", f"_goal_{i}_side.png")
        goal_side_path = grab_side_image_path(env, tmp_top_goal, tmp_side_goal)

        # Action = goal - observation (x,y) only
        delta = goal_xy - obs_xy
        dx_str = f"{delta[0]:.2f}"
        dy_str = f"{delta[1]:.2f}"

        obs_b64 = encode_image(obs_side_path)
        goal_b64 = encode_image(goal_side_path)

        if not save_debug_images:
            for _p in (obs_side_path, goal_side_path):
                try:
                    os.remove(_p)
                except OSError:
                    pass

        system_message = {"role": "system", "content": system_prompt}
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "observation:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}"}},
                {"type": "text", "text": "goal:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{goal_b64}"}},
            ],
        }
        assistant_message = {"role": "assistant", "content": f"action:[{dx_str}, {dy_str}]"}

        record = {"messages": [system_message, user_message, assistant_message]}

        jf.write(json.dumps(record) + "\n")
        csv_writer.writerow([
            system_prompt,
            "observation",
            f"data:image/png;base64,{obs_b64}",
            "goal",
            f"data:image/png;base64,{goal_b64}",
            assistant_message["content"],
        ])

if save_debug_images:
    print(f"   - Debug image dir: {debug_dir}")
