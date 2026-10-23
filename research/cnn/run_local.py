"""Run the whole F127 pipeline on one machine (e.g. a local GPU). Usage, from the repo root:

    python research/cnn/run_local.py [--skip-fetch]

Same pre-registered design as the Actions workflow, except training is not time-capped
(early stopping / 20 epochs only), which is closer to the paper.
"""
import os
import subprocess
import sys

env = dict(os.environ, TRAIN_BUDGET_MIN="100000", PYTHONIOENCODING="utf-8")


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, env=env)


if "--skip-fetch" not in sys.argv:
    run("research/cnn/fetch_long.py")
for seed in range(1, 6):
    if not os.path.exists(f"research/cnn/work/model_{seed}.pt"):
        run("research/cnn/train.py", str(seed))
    if not os.path.exists(f"research/cnn/work/pred_{seed}_0.parquet"):
        run("research/cnn/predict.py", str(seed), "0", "1")
with open("research/cnn/out_local.txt", "w", encoding="utf-8") as f:
    subprocess.run([sys.executable, "research/cnn/evaluate_cnn.py"], check=True, env=env, stdout=f)
print(open("research/cnn/out_local.txt", encoding="utf-8").read())
