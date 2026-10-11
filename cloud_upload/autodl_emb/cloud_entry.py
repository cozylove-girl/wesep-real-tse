"""Run the upstream WeSep trainer on cloud CUDA, with timing wrappers.

Benchmark updates a disposable model, never the formal experiment checkpoint.
This script refuses non-Linux hosts before importing torch.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from cloud_common import save_json
from time_estimate import describe


class BenchmarkComplete(Exception):
    pass


class TimedLoader:
    """Measure actual DataLoader wait + complete optimization, including the last step."""
    def __init__(self, loader, *, limit, warmup, torch_module):
        self.loader, self.limit, self.warmup, self.torch = loader, limit, warmup, torch_module
        self.times = []
        self.expanded_target_batch = None
        self.generator = None

    def __iter__(self):
        self.generator = self._iterate()
        return self.generator

    def _iterate(self):
        start = time.perf_counter()
        iterator = iter(self.loader)
        pending = None
        try:
            for _ in range(self.limit):
                batch = next(iterator)
                pending = start
                if self.expanded_target_batch is None:
                    self.expanded_target_batch = int(batch["wav_mix"].shape[0])
                yield batch
                self.torch.cuda.synchronize()
                end = time.perf_counter()
                self.times.append(end-pending)
                pending = None
                start = end
        finally:
            # The upstream executor breaks before requesting another item.
            if pending is not None:
                self.torch.cuda.synchronize()
                self.times.append(time.perf_counter()-pending)

    def finish(self):
        if self.generator is not None:
            self.generator.close()
        if len(self.times) <= self.warmup:
            raise RuntimeError("Not enough completed timing steps after warmup")
        return {"timing": describe(self.times[self.warmup:]), "warmup_steps": self.warmup,
                "completed_steps": len(self.times), "first_step_seconds": self.times[0],
                "expanded_target_batch_per_gpu": self.expanded_target_batch}


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Cloud CUDA training only; do not run this entry on the local Windows computer.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measure", type=int, default=200)
    parser.add_argument("--validation-measure", type=int, default=30)
    args = parser.parse_args()
    work = args.work_root.resolve()
    sys.path[:0] = [str(work/"runtime/wesep-real-tse"), str(work/"runtime/wespeaker")]
    import torch
    import yaml
    import wesep.bin.train as upstream
    from wesep.utils.executor import Executor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    original_train, original_cv = Executor.train, Executor.cv
    rank = int(os.environ.get("RANK", "0"))
    rank_results = {"rank": rank, "gpu": torch.cuda.get_device_name(rank), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "batch_size_mixtures": config["dataloader_args"]["batch_size"]}

    def reduce_loss(value):
        if not math.isfinite(float(value)):
            raise RuntimeError("Nonfinite training/validation loss")
        tensor = torch.tensor(float(value), device=torch.device("cuda"))
        torch.distributed.all_reduce(tensor)
        return float(tensor.item()/torch.distributed.get_world_size())

    def wrapped_train(self, *positional, **keyword):
        positional = list(positional)
        model = positional[1][0]
        raw = model.module if hasattr(model, "module") else model
        if any(p.requires_grad for p in raw.spk_ft.encoder.spk_model.parameters()):
            raise RuntimeError("ECAPA parameters should remain frozen")
        if bool(config["model_args"]["tse_model"]["separator"]["causal"]):
            raise RuntimeError("Requested EMB baseline is offline, not causal")
        epoch = int(keyword["epoch"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        timer = None
        if args.benchmark:
            limit = args.warmup+args.measure
            positional[2] = limit
            timer = TimedLoader(positional[0], limit=limit, warmup=args.warmup, torch_module=torch)
            positional[0] = timer
        result = original_train(self, *positional, **keyword)
        torch.cuda.synchronize()
        loss = reduce_loss(result[0])
        elapsed = time.perf_counter()-started
        if args.benchmark:
            stats = timer.finish()
            gradients = [p.grad for p in raw.sep_model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("Separator gradients missing/nonfinite")
            if stats["expanded_target_batch_per_gpu"] != 2*config["dataloader_args"]["batch_size"]:
                raise RuntimeError("Collate target expansion does not match two-speaker recipe")
            rank_results["train"] = {**stats, "loss": loss, "epoch_elapsed_seconds": elapsed,
                                     "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                                     "separator_gradient_check": "passed", "ecapa_frozen": True}
        else:
            path = work/"timing"/f"epoch_{epoch:03d}_rank{rank}.json"
            save_json(path, {"epoch": epoch, "rank": rank, "train_seconds": elapsed,
                             "train_loss": loss, "train_steps": int(positional[2])})
        return loss, result[1]

    def wrapped_cv(self, *positional, **keyword):
        positional = list(positional)
        epoch = int(keyword["epoch"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        timer = None
        if args.benchmark:
            limit = min(args.validation_measure+3, int(positional[2]))
            positional[2] = limit
            timer = TimedLoader(positional[0], limit=limit, warmup=min(3, limit-1), torch_module=torch)
            positional[0] = timer
        result = original_cv(self, *positional, **keyword)
        torch.cuda.synchronize()
        loss = reduce_loss(result[0])
        elapsed = time.perf_counter()-started
        if args.benchmark:
            rank_results["validation"] = {**timer.finish(), "loss": loss, "elapsed_seconds": elapsed}
            save_json(work/"benchmark"/f"rank{rank}.json", rank_results)
            raise BenchmarkComplete()
        else:
            path = work/"timing"/f"epoch_{epoch:03d}_rank{rank}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update({"validation_seconds": elapsed, "validation_loss": loss,
                           "validation_steps": int(positional[2]), "total_seconds": record["train_seconds"]+elapsed})
            save_json(path, record)
            if rank == 0:
                records = [json.loads(p.read_text()) for p in sorted((work/"timing").glob("epoch_*_rank0.json"))
                           if "total_seconds" in json.loads(p.read_text()) and json.loads(p.read_text())["epoch"] <= epoch]
                recent = records[-3:]
                average = sum(r["total_seconds"] for r in recent)/len(recent)
                remaining = max(0, config["num_epochs"]-epoch)
                save_json(work/"live_time_estimate.json", {"completed_epoch": epoch, "recent_mean_epoch_seconds": average,
                           "estimated_remaining_training_validation_hours": average*remaining/3600,
                           "based_on_recent_epochs": [r["epoch"] for r in recent],
                           "notice": "Epoch timings include loader startup; final evaluation and checkpoint I/O excluded."})
                print(f"[ETA] epoch {epoch}/{config['num_epochs']}，预计剩余训练+验证 {average*remaining/3600:.2f} h", flush=True)
        return loss, result[1]

    Executor.train, Executor.cv = wrapped_train, wrapped_cv
    try:
        upstream.train(config=str(args.config))
        if args.benchmark:
            raise RuntimeError("Benchmark should terminate after disposable validation")
    except BenchmarkComplete:
        torch.distributed.barrier()
        if rank == 0:
            print("Disposable benchmark finished; formal model initialization unchanged.", flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    if platform.system() != "Linux":
        raise RuntimeError("Cloud Linux CUDA only; do not run training on local Windows.")
    from torch.distributed.elastic.multiprocessing.errors import record
    record(main)()
