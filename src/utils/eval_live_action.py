
import os, sys, json, re, base64, math, random
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.environments.cube_environment import CubeEnvironment

client = OpenAI(api_key=apikey)

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


def run_trial(env: CubeEnvironment):
    # Sample observation & goal positions
    obs_xy = [round(random.uniform(0.0, 0.8), 2), round(random.uniform(0.0, 0.8), 2)]
    env.reset(start_pos=[obs_xy[0], obs_xy[1], 0.02])
    tmp_top_obs = "_obs_top.png"; tmp_side_obs = "_obs_side.png"
    side_obs_path = grab_side_image(env, tmp_top_obs, tmp_side_obs)

    goal_xy = [round(random.uniform(0.0, 0.8), 2), round(random.uniform(0.0, 0.8), 2)]
    env.reset(start_pos=[goal_xy[0], goal_xy[1], 0.02])
    tmp_top_goal = "_goal_top.png"; tmp_side_goal = "_goal_side.png"
    side_goal_path = grab_side_image(env, tmp_top_goal, tmp_side_goal)

    obs_b64 = encode_image(side_obs_path)
    goal_b64 = encode_image(side_goal_path)

    # cleanup side images
    for pth in (side_obs_path, side_goal_path):
        try: os.remove(pth)
        except OSError: pass

    true_dx = round(goal_xy[0] - obs_xy[0], 2)
    true_dy = round(goal_xy[1] - obs_xy[1], 2)

    messages = build_messages(obs_b64, goal_b64)
    model_answer = call_model(messages, model="ft:gpt-4o-2024-08-06:personal:cube-env:C2lSpZJd")
    pred_dx, pred_dy = parse_action(model_answer)

    err_dx = pred_dx - true_dx
    err_dy = pred_dy - true_dy
    l2 = math.sqrt(err_dx**2 + err_dy**2)

    return {
        "obs": obs_xy,
        "goal": goal_xy,
        "true": [true_dx, true_dy],
        "pred_raw": model_answer,
        "pred": [pred_dx, pred_dy],
        "error": [err_dx, err_dy],
        "l2": l2,
    }

def run_env(env):
    goal_path = "data/cube_goal_left.png"
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

def main():
    env = CubeEnvironment(gui=True,)
    results = []
    trials = 1
    for i in range(trials):
        r = run_trial(env)
        results.append(r)
        print(f"Trial {i+1}/{trials} obs={r['obs']} goal={r['goal']} true={r['true']} pred={r['pred']} l2={r['l2']:.3f}")
    env.close()

    avg_l2 = sum(r['l2'] for r in results) / len(results)
    print(f"Average L2 error over {len(results)} trials: {avg_l2:.4f}")

    return 0

def main_run_env():
    env = CubeEnvironment(gui=True, goal_position=[0.3, 0.65, 0])
    run_env(env)

if __name__ == '__main__':
    sys.exit(main_run_env())
