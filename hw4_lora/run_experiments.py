import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from data import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--max-gpu-seconds", type=float, default=7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = json.loads(Path("experiments.json").read_text())
    pending = list(args.cases or config["experiments"])
    args.output.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output / "resource_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    spent = sum(row["gpu_seconds"] for row in ledger)
    active = {}
    leases = {}
    devices = {}
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    for line in query.splitlines():
        index, uuid, memory = [value.strip() for value in line.split(",")]
        if int(index) in args.gpus:
            if int(memory) > 1024:
                raise RuntimeError(f"GPU {index} is occupied: {memory} MiB")
            devices[int(index)] = uuid
    lease_dir = Path.home() / ".gpu-leases"
    lease_dir.mkdir(exist_ok=True)
    for index in args.gpus:
        lease = (lease_dir / f"{socket.gethostname()}-{devices[index]}.lock").open("a")
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases[index] = lease
    print(f"Leased {devices}; prior GPU seconds={spent:.1f}", flush=True)
    budget_exceeded = False
    while pending or active:
        now = time.monotonic()
        used = spent + sum(now - job["start"] for job in active.values())
        if used >= args.max_gpu_seconds:
            budget_exceeded = True
            pending.clear()
            for job in active.values():
                if job["process"].poll() is None:
                    job["process"].terminate()
        for index, job in list(active.items()):
            code = job["process"].poll()
            if code is None:
                continue
            seconds = time.monotonic() - job["start"]
            spent += seconds
            job["log"].close()
            record = {
                "name": job["name"], "case": job["case"], "host": socket.gethostname(),
                "gpu_index": index, "gpu_uuid": devices[index], "pid": job["process"].pid,
                "started_utc": job["utc"], "gpu_seconds": seconds, "exit_code": code,
                "budget_exceeded": budget_exceeded, "smoke": args.smoke,
            }
            ledger.append(record)
            write_json(ledger_path, ledger)
            print(json.dumps(record), flush=True)
            del active[index]
        if not budget_exceeded:
            for index in args.gpus:
                if index in active or not pending:
                    continue
                case = pending.pop(0)
                name = f"smoke_{case}" if args.smoke else case
                if (args.output / name / "summary.json").exists():
                    print(f"Skip completed {name}", flush=True)
                    continue
                command = [sys.executable, "train.py", "--case", case, "--name", name, "--output", str(args.output)]
                if args.smoke:
                    command.extend(["--max-steps", "1", "--train-limit", "32", "--eval-limit", "16"])
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index], OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
                log = (args.output / f"{name}.log").open("w")
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                active[index] = {"process": process, "start": time.monotonic(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "case": case, "name": name, "log": log}
                print(f"Started {name} on GPU {index}, pid={process.pid}", flush=True)
        if pending or active:
            time.sleep(1)
    if budget_exceeded or any(row["exit_code"] != 0 for row in ledger):
        raise SystemExit("Budget exceeded or a run failed; inspect resource_ledger.json")


if __name__ == "__main__":
    main()
