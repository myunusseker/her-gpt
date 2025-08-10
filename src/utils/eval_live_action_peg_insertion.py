import os, sys, json, re, base64, math, random
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.environments.peg_insertion_environment import PegInsertionEnvironment

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = (
    "You are a robotic agent that produces actions for a peg insertion task based on observation images.\n"
    "Input: You will be given an observation.\n"
    "Output format: action:[param1, param2], each formatted with exactly two decimals; no extra text.\n"
)

ACTION_RE = re.compile(r"action:\[(-?\d+\.\d{2}),\s*(-?\d+\.\d{2})]\s*$", re.I)


def encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def grab_side_image(env: PegInsertionEnvironment, wrist_tmp: str, side_tmp: str) -> str:
    """Render both views; return side image path (caller encodes)."""
    env.render_views(save_images=True, wrist_path=wrist_tmp, side_path=side_tmp)
    # we only need side view for the model
    try:
        os.remove(wrist_tmp)
    except OSError:
        pass
    return side_tmp


def build_messages(obs_b64: str):
    """Build messages for peg insertion task (observation only, no goal image)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "observation:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}"}},
        ]},
    ]


def parse_action(text):
    m = ACTION_RE.search(text.strip())
    if not m:
        # fallback: extract first two floats
        nums = re.findall(r"-?\d+\.\d+", text)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        raise ValueError(f"Cannot parse action from: {text[:120]}")
    return float(m.group(1)), float(m.group(2))


def call_model(messages, model):
    # Chat Completions (vision) style
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def run_peg_insertion_env(env, model_name="gpt-4o"):
    """Run interactive peg insertion evaluation."""
    print("Starting peg insertion evaluation...")
    print("The goal is to insert the green peg into the red hole.")
    print("The model will predict corrections based on visual observations.")

    offset_xy = [0.02, 0.015]
    # Apply random initial offset to create misalignment
    
    for i in range(10):
        print(f"\n--- Trial {i+1}/10 ---")
        
        # Reset environment to random offset position
        env.reset()
        env.apply_action(offset=np.array([offset_xy[0], offset_xy[1], 0.0]))

        
        if input("Continue with this trial? Y|n: ").lower() == "n":
            break
            
        # Get current observation
        side_obs_path = grab_side_image(env, "_obs_wrist.png", "_obs_side.png")
        obs_b64 = encode_image(side_obs_path)
        
        # Get model prediction
        messages = build_messages(obs_b64)
        print("Getting model prediction...")
        model_answer = call_model(messages, model=model_name)
        print(f"Model response: {model_answer}")
        
        pred_dx, pred_dy = parse_action(model_answer)
        print(f"Predicted correction: dx={pred_dx:.2f}, dy={pred_dy:.2f}")
        offset_xy = [offset_xy[0]+pred_dx/100., offset_xy[1]+pred_dy/100.]
        # Clean up temporary files
        try:
            os.remove(side_obs_path)
        except OSError:
            pass
                
    env.close()
    print("Evaluation completed!")


def main_run_peg_insertion():
    """Main function to run peg insertion evaluation."""
    # You can change these parameters
    use_gui = True  # Set to False for headless mode
    
    env = PegInsertionEnvironment(gui=use_gui)
    
    try:
        run_peg_insertion_env(env, model_name="ft:gpt-4o-2024-08-06:personal:peg-insertion:C2s2zBGw")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        env.close()


if __name__ == '__main__':
    import numpy as np  # Import numpy for random offset generation
    sys.exit(main_run_peg_insertion())
