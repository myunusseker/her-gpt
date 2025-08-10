import json
import csv
import os
import base64
import numpy as np

from src.environments.peg_insertion_environment import PegInsertionEnvironment

# Settings
n_samples = 100  # number of (observation, goal) pairs
dataset_name = "peg_insertion_100_train"
out_jsonl = f"{dataset_name}.jsonl"
out_csv = f"{dataset_name}.csv"
save_debug_images = True  # only set True if you still want PNGs on disk for inspection
debug_dir = f"data/{dataset_name}_debug"
if save_debug_images:
    os.makedirs(debug_dir, exist_ok=True)

system_prompt_old = "You are a robotic agent that produces actions to insert the green peg into the red hole based on observation images."
system_prompt = (
    "You are a robotic agent that produces actions for a peg insertion task based on observation images.\n"
    "Input: You will be given an observation.\n"
    "Output format: action:[param1, param2], each formatted with exactly two decimals; no extra text.\n"
)

def grab_side_image_path(env, wrist_path_tmp, side_path_tmp):
    """Render both views; return side image path (caller encodes)."""
    env.render_views(wrist_path=wrist_path_tmp, side_path=side_path_tmp)
    # Optionally remove the wrist image if not debugging
    if not save_debug_images:
        try:
            os.remove(wrist_path_tmp)
        except OSError:
            pass
    return side_path_tmp

# Official image encode helper
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

env = PegInsertionEnvironment(gui=False)

with open(out_jsonl, "w") as jf, open(out_csv, "w", newline="") as cf:
    csv_writer = csv.writer(cf)
    csv_writer.writerow([
        "system_prompt",
        "input_text_observation",
        "input_image_observation",
        "output_text",
    ])

    for i in range(n_samples):
        # Observation spawn
        obs_xy = np.round(np.random.uniform(-3.00, 3.00, size=2), 2)
        env.reset()
        env.apply_action(offset=np.array([obs_xy[0]/100., obs_xy[1]/100., 0.0]))    
        tmp_wrist_obs = os.path.join(debug_dir if save_debug_images else ".", f"_obs_{i}_wrist.png")
        tmp_side_obs = os.path.join(debug_dir if save_debug_images else ".", f"_obs_{i}_side.png")
        obs_side_path = grab_side_image_path(env, tmp_wrist_obs, tmp_side_obs)

        # Action = goal - observation (x,y) only
        delta = -obs_xy
        dx_str = f"{delta[0]:.2f}"
        dy_str = f"{delta[1]:.2f}"

        obs_b64 = encode_image(obs_side_path)
        
        if not save_debug_images:
            try:
                os.remove(tmp_wrist_obs)
            except OSError:
                pass

        system_message = {"role": "system", "content": system_prompt}
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "observation:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}"}},
            ],
        }
        assistant_message = {"role": "assistant", "content": f"action:[{dx_str}, {dy_str}]"}

        record = {
            "messages": [
                system_message, 
                user_message, 
                assistant_message
            ]
        }

        jf.write(json.dumps(record) + "\n")
        csv_writer.writerow([
            system_prompt,
            "observation",
            f"data:image/png;base64,{obs_b64}",
            assistant_message["content"],
        ])

if save_debug_images:
    print(f"   - Debug image dir: {debug_dir}")
