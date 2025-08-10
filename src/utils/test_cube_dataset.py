import json, re, sys, os

# Usage: python test_cube_dataset.py path/to/dataset.jsonl
# Validates action format: action:[dx, dy] with two decimals each and value range within [-1.0, 1.0]

ACTION_RE = re.compile(r"^action:\[(-?\d+\.\d{2}),\s*(-?\d+\.\d{2})]$")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_cube_dataset.py dataset.jsonl")
        return 1
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return 1

    n = 0
    bad = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"Line {n} JSON error: {e}")
                bad += 1
                continue
            msgs = rec.get('messages') or []
            if not msgs or msgs[-1].get('role') != 'assistant':
                print(f"Line {n} missing assistant message")
                bad += 1
                continue
            content = msgs[-1].get('content')
            if not isinstance(content, str):
                print(f"Line {n} assistant content not string")
                bad += 1
                continue
            m = ACTION_RE.match(content)
            if not m:
                print(f"Line {n} bad action format: {content}")
                bad += 1
                continue
            dx = float(m.group(1))
            dy = float(m.group(2))
            # Range sanity check (spawn range 0..0.8 so delta in [-0.8,0.8]) with tiny slack
            if not (-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0):
                print(f"Line {n} action out of range: ({dx},{dy})")
                bad += 1
    print(f"Checked {n} records; format failures: {bad}")
    return 0 if bad == 0 else 2

if __name__ == '__main__':
    raise SystemExit(main())
