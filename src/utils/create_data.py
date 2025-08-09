import json
import csv
import numpy as np

# Settings
mass = 2.0
n_samples = 100
jsonl_file = "kick_to_goal_100_eval.jsonl"
csv_file = "kick_to_goal_100_eval.csv"

def compute_force(state, goal, mass):
    return (goal - state) * mass

system_prompt = "You're a game character who kicks a cube toward the goal."

# Write to JSONL and CSV
with open(jsonl_file, "w") as jf, open(csv_file, "w", newline='') as cf:
    csv_writer = csv.writer(cf)
    csv_writer.writerow(["system_prompt", "user_message", "assistant_response"])

    for _ in range(n_samples):
        state = np.round(np.random.uniform(-5, 5, size=2), 2)
        goal = np.round(np.random.uniform(-5, 5, size=2), 2)
        force = np.round(compute_force(state, goal, mass), 2)

        user_msg = f"Current position: {list(state)}, Goal: {list(goal)}"
        assistant_msg = f"Force: {list(force)}"

        # Write JSONL
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        jf.write(json.dumps(record) + "\n")

        # Write CSV
        csv_writer.writerow([system_prompt, user_msg, assistant_msg])

print(f"✅ Saved {n_samples} examples to:")
print(f"   - JSONL: {jsonl_file}")
print(f"   - CSV:   {csv_file}")
