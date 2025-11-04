#!/usr/bin/env python3
# run_grid_improved.py
import argparse
import subprocess
import sys
import os
import time
import datetime
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

LOCK_WAIT = 0.5

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def acquire_lock(lock_path: Path, wait_interval=LOCK_WAIT, timeout=600):
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - start > timeout:
                return False
            time.sleep(wait_interval)

def release_lock(lock_path: Path):
    try:
        lock_path.unlink()
    except Exception:
        pass

def run_cmd(cmd, cwd=None, env=None, capture=True):
    if capture:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, text=True)
        return proc.returncode, proc.stdout
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=env)
        return proc.returncode, None

def generate_dataset_if_needed(m: int, data_file: Path, python_exec: str, 
                               seed: int, force_regen: bool, runlog_file: Path):
    data_file.parent.mkdir(parents=True, exist_ok=True)
    
    if data_file.exists() and not force_regen:
        with runlog_file.open("a", encoding="utf-8") as f:
            f.write(f"[DATA] exists, skipping: {data_file}\n")
        return True, "exists"
    
    lock_path = data_file.with_suffix(data_file.suffix + ".lock")
    got_lock = acquire_lock(lock_path)
    
    if not got_lock:
        with runlog_file.open("a", encoding="utf-8") as f:
            f.write(f"[DATA][ERROR] lock timeout for {data_file}\n")
        return False, "lock_error"
    
    try:
        if data_file.exists() and not force_regen:
            return True, "created_by_other"
        
        cmd = [python_exec, "generate_data_new.py",
               "--num_samples", str(m),
               "--output_path", str(data_file),
               "--seed", str(seed)]
        
        with runlog_file.open("a", encoding="utf-8") as f:
            f.write(f"[DATA][CMD] {' '.join(cmd)} at {datetime.datetime.now().isoformat()}\n")
            f.flush()
        
        rc, out = run_cmd(cmd, capture=True)
        
        with runlog_file.open("a", encoding="utf-8") as f:
            f.write(out + "\n")
            f.write(f"[DATA] rc={rc}\n")
        
        if rc != 0:
            return False, f"gen_failed_rc_{rc}"
        
        return True, "generated"
    
    finally:
        release_lock(lock_path)

def run_single_experiment(n: int, m: int, args, runlog_lock, top_runlog_path: Path):
    """Run a single (N, M) experiment."""
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    data_file = data_dir / f"m_{m}.npz"
    outdir = results_dir / f"n_{n}_m_{m}"
    outdir.mkdir(parents=True, exist_ok=True)
    per_run_log = outdir / "run.log"
    
    metrics_path = outdir / "metrics.json"
    if metrics_path.exists() and args.skip_completed and not args.force_rerun:
        summary = {"n": n, "m": m, "status": "skipped_existing", "outdir": str(outdir)}
        with runlog_lock:
            with top_runlog_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        return summary
    
    # Generate data
    gen_ok, gen_msg = generate_dataset_if_needed(m, data_file, args.python, 
                                                 args.seed, args.force_regen, 
                                                 top_runlog_path)
    if not gen_ok:
        summary = {"n": n, "m": m, "status": "gen_failed", "reason": gen_msg, 
                  "outdir": str(outdir)}
        with runlog_lock:
            with top_runlog_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        return summary
    
    # Get dataset checksum
    try:
        sha = sha256_of_file(data_file)
    except Exception:
        sha = None
    
    # Build training command
    train_cmd = [
        args.python, "check_new_fixed.py",  # Use the fixed version
        "--num_views", str(n),
        "--data_path", str(data_file),
        "--outdir", str(outdir),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--mask_ratio", str(args.mask_ratio),
        "--embed_dim", str(args.embed_dim),
        "--num_layers", str(args.num_layers),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
    ]
    
    # Add memory efficiency flags based on N
    if n > 5 or args.always_memory_efficient:
        train_cmd.append("--memory_efficient")
    elif n > 3 and args.use_accumulation:
        train_cmd.extend(["--accumulation_steps", str(args.accumulation_steps)])
    
    if args.auto_adjust_batch:
        train_cmd.append("--auto_adjust_batch")
    
    # Run training
    with per_run_log.open("w", encoding="utf-8") as prunef:
        prunef.write(f"[RUN] n={n} m={m} at {datetime.datetime.now().isoformat()}\n")
        prunef.write(f"[RUN][CMD] {' '.join(train_cmd)}\n")
        prunef.flush()
        
        start_time = time.time()
        rc, out = run_cmd(train_cmd, capture=True)
        elapsed = time.time() - start_time
        
        prunef.write(out + "\n")
        prunef.write(f"[RUN] rc={rc}, elapsed={elapsed:.2f}s\n")
        prunef.flush()
    
    status = "ok" if rc == 0 else "train_failed"
    summary = {
        "n": n, "m": m, "status": status, "rc": rc, "outdir": str(outdir),
        "dataset_sha256": sha, "elapsed_sec": elapsed,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    with runlog_lock:
        with top_runlog_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    
    return summary

def main():
    p = argparse.ArgumentParser(description="Grid search over N (num_views) and M (dataset_size)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--results_dir", type=str, default="results")
    
    # Better default M values (geometric spacing)
    p.add_argument("--m_list", nargs="+", type=int,
                   default=[10000, 5000, 2000, 1000, 500, 200, 100])
    
    # Extended N range for ablation
    p.add_argument("--n_list", nargs="+", type=int, 
                   default=[1, 2, 3, 4, 5, 8, 10, 15, 20])
    
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--force_regen", action="store_true")
    p.add_argument("--force_rerun", action="store_true")
    p.add_argument("--skip_completed", action="store_true", 
                   help="Skip if metrics.json exists (safe resume)")
    p.add_argument("--max_workers", type=int, default=1)
    
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=25)
    
    # Memory management
    p.add_argument("--always_memory_efficient", action="store_true",
                   help="Always use sequential processing (slowest, safest)")
    p.add_argument("--use_accumulation", action="store_true",
                   help="Use gradient accumulation for N>3")
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--auto_adjust_batch", action="store_true",
                   help="Auto-adjust batch size based on N")
    
    args = p.parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run log
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    top_runlog_path = results_dir / f"runlog_{timestamp}.log"
    
    config = {
        "seed": args.seed,
        "m_list": args.m_list,
        "n_list": args.n_list,
        "started": datetime.datetime.now().isoformat(),
        "memory_strategy": "always_efficient" if args.always_memory_efficient 
                          else ("accumulation" if args.use_accumulation else "auto"),
    }
    
    with top_runlog_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    
    # Create task list: iterate N (outer) × M (inner)
    tasks = [(n, m) for n in args.n_list for m in args.m_list]
    
    print(f"Running {len(tasks)} experiments: N={args.n_list}, M={args.m_list}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Memory strategy: {config['memory_strategy']}")
    
    runlog_lock = threading.Lock()
    results = []
    
    if args.max_workers == 1:
        # Sequential execution
        for i, (n, m) in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Running N={n}, M={m}...")
            summary = run_single_experiment(n, m, args, runlog_lock, top_runlog_path)
            results.append(summary)
            print(f"  Status: {summary['status']}")
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_to_task = {
                ex.submit(run_single_experiment, n, m, args, runlog_lock, top_runlog_path): (n, m)
                for (n, m) in tasks
            }
            
            completed = 0
            for fut in as_completed(future_to_task):
                (n, m) = future_to_task[fut]
                completed += 1
                try:
                    summary = fut.result()
                    print(f"[{completed}/{len(tasks)}] N={n}, M={m}: {summary['status']}")
                except Exception as e:
                    summary = {"n": n, "m": m, "status": "exception", "error": str(e)}
                    with runlog_lock:
                        with top_runlog_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(summary) + "\n")
                results.append(summary)
    
    # Save summary
    summary_path = results_dir / f"sweep_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "completed_at": datetime.datetime.now().isoformat(),
            "config": config,
            "results": results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Sweep finished!")
    print(f"Summary: {summary_path}")
    print(f"Runlog: {top_runlog_path}")
    
    # Quick stats
    success = sum(1 for r in results if r.get('status') == 'ok')
    failed = sum(1 for r in results if r.get('status') == 'train_failed')
    skipped = sum(1 for r in results if r.get('status') == 'skipped_existing')
    
    print(f"Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()