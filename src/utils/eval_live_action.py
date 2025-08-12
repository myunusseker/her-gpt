
import os, sys, json, re, base64, math, random
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.environments.cube_environment import CubeEnvironment

client = OpenAI(api_key="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")

system_prompt = (
    "You are a robotic agent that produces actions to move a cube based on observation and goal images.\n"
    "Input: You will be given an observation image and a goal image.\n"
    "Output format: action:[param1, param2], each formatted with exactly two decimals; no extra text.\n"
)

ACTION_RE = re.compile(r"action:\[(-?\d+\.\d{2}),\s*(-?\d+\.\d{2})]\s*$", re.I)


def encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def grab_side_image(env: CubeEnvironment, top_tmp: str, side_tmp: str) -> str:
    env.render_views(topdown_path=top_tmp, side_path=side_tmp, use_static_side=True)
    # we only need side view
    try:
        os.remove(top_tmp)
    except OSError:
        pass
    return side_tmp


def build_messages(obs_b64: str, goal_b64: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "observation"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}"}},
            {"type": "text", "text": "goal"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{goal_b64}"}},
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


def run_env(env):
    goal_path = "data/cube_goal.png"
    position = [0.0, 0.0]
    for i in range(10):
        env.reset(start_pos=[position[0], position[1], 0.02])
        if input("Continue? Y|n") == "n":
            break
        side_obs_path = grab_side_image(env, "_obs_top.png", "_obs_side.png")

        obs_b64 = encode_image(side_obs_path)
        goal_b64 = encode_image(goal_path)

        messages = build_messages(obs_b64, goal_b64)
        model_answer = call_model(messages, model="ft:gpt-4o-2024-08-06:personal:cube-env-100:C2mqsimT")
        pred_dx, pred_dy = parse_action(model_answer)
        position[0] += pred_dx
        position[1] += pred_dy
        print(f"predicted action: dx={pred_dx}, dy={pred_dy}")
    env.close()

def main_run_env():
    env = CubeEnvironment(gui=True)
    run_env(env)

if __name__ == '__main__':
    sys.exit(main_run_env())
