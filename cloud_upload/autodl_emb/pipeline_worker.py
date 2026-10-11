"""EMB cloud stages. Never execute training on the local Windows computer."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from cloud_common import PACKAGE, REPO, save_json, sha256
from time_estimate import estimate


def config(settings):
    import yaml
    work = Path(settings["work_root"])
    return yaml.safe_load((work/"prepared/train.yaml").read_text(encoding="utf-8"))


def launch(settings, cfg_path, extra=None):
    from gpu_monitor import GPUMemoryMonitor
    work = Path(settings["work_root"])
    count = len(settings["gpu_ids"].split(","))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={count}",
               str(PACKAGE/"cloud_entry.py"), "--config", str(cfg_path), "--work-root", str(work)]
    command.extend(extra or [])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(work/"runtime/wesep-real-tse"), str(work/"runtime/wespeaker"), str(PACKAGE)])
    env["MPLBACKEND"] = "Agg"
    report_file = work/"benchmark/nvidia_memory.json" if extra and "--benchmark" in extra else work/"nvidia_memory_training.json"
    with GPUMemoryMonitor(settings["gpu_ids"], report_file):
        subprocess.run(command, cwd=work/"runtime/wesep-real-tse", env=env, check=True)


def benchmark(settings):
    import yaml
    work = Path(settings["work_root"])
    cfg = config(settings)
    cfg["exp_dir"] = str(work/"benchmark/disposable_exp")
    cfg.pop("checkpoint", None)
    destination = work/"benchmark/config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    previous = work/"time_estimate.json"
    if previous.exists():
        previous.replace(work/"benchmark/previous_time_estimate.json")
    launch(settings, destination, ["--benchmark", "--warmup", str(settings["benchmark_warmup"]),
                                  "--measure", str(settings["benchmark_steps"]),
                                  "--validation-measure", str(settings["benchmark_val_steps"])])
    world = len(settings["gpu_ids"].split(","))
    ranks = [json.loads((work/"benchmark"/f"rank{r}.json").read_text()) for r in range(world)]
    train_seconds = max(r["train"]["timing"]["mean_seconds"] for r in ranks)
    val_seconds = max(r["validation"]["timing"]["mean_seconds"] for r in ranks)
    report = json.loads((work/"prepared/data_report.json").read_text())
    result = estimate(train_pairs=report["train-100"]["mixture_records"], val_pairs=report["dev"]["mixture_records"],
                      mixtures_per_batch=settings["batch_size"], world_size=world, epochs=settings["epochs"],
                      train_seconds=train_seconds, val_seconds=val_seconds)
    result.update({"status": "measured_cloud_benchmark", "config_sha256": sha256(work/"prepared/train.yaml"),
                   "gpu_reports": ranks, "notice": "Disposable benchmark is excluded from formal training; per-step CUDA synchronization can make this estimate conservative. First epoch ETA is more representative."})
    monitor_file = work/"benchmark/nvidia_memory.json"
    if monitor_file.is_file():
        result["nvidia_memory_monitor"] = json.loads(monitor_file.read_text(encoding="utf-8"))
    save_json(work/"time_estimate.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    budget = settings.get("max_estimated_train_hours", 0)
    if budget > 0 and result["train_and_validation_hours"] > budget:
        raise RuntimeError(f"估算训练+验证 {result['train_and_validation_hours']:.2f} h 超过你设置的 {budget} h；未开始正式训练。")


def train(settings):
    import yaml
    work = Path(settings["work_root"])
    marker = work/"time_estimate.json"
    if not marker.exists() or json.loads(marker.read_text())["config_sha256"] != sha256(work/"prepared/train.yaml"):
        raise RuntimeError("请先通过同一配置的 CUDA 训练 benchmark")
    cfg = config(settings)
    model_dir = work/"exp/emb/models"
    checkpoints = sorted(model_dir.glob("checkpoint_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    last_epoch = 0
    if checkpoints:
        latest = checkpoints[-1]
        last_epoch = int(latest.stem.split("_")[-1])
        cfg["checkpoint"] = str(latest)
        print(f"恢复已保存的 epoch {last_epoch}；未保存的轮次会重跑，非逐 batch 精确续训。", flush=True)
    destination = work/"prepared/train_run.yaml"
    destination.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if last_epoch < settings["epochs"]:
        launch(settings, destination)
    final = model_dir/f"checkpoint_{settings['epochs']}.pt"
    if not final.is_file():
        raise RuntimeError("缺少最终 epoch checkpoint，不能标记训练完成")
    save_json(work/"training_complete.json", {"status": "complete", "epochs": settings["epochs"],
                                             "checkpoint": str(final), "sha256": sha256(final)})


def average(settings):
    import torch
    work = Path(settings["work_root"])
    if not (work/"training_complete.json").exists():
        raise RuntimeError("Formal training incomplete")
    cfg = config(settings)
    model_dir = work/"exp/emb/models"
    choices = []
    for path in model_dir.glob("checkpoint_[0-9]*.pt"):
        epoch = int(path.stem.split("_")[-1])
        timing = work/"timing"/f"epoch_{epoch:03d}_rank0.json"
        if timing.exists():
            record = json.loads(timing.read_text())
            if "validation_loss" in record and math.isfinite(record["validation_loss"]):
                choices.append((float(record["validation_loss"]), epoch, path))
    keep = min(int(cfg.get("num_avg", 10)), len(choices))
    if keep == 0:
        raise RuntimeError("No checkpoints with recorded validation loss")
    selected = sorted(choices)[:keep]
    total = None
    for _, epoch, path in selected:
        weights = torch.load(path, map_location="cpu", weights_only=False)["models"][0]
        if total is None:
            total = {key: value.clone() for key, value in weights.items()}
        else:
            if total.keys() != weights.keys():
                raise RuntimeError("Checkpoint state dictionaries differ")
            for key in total:
                if total[key].is_floating_point() or total[key].is_complex():
                    total[key] += weights[key]
                else:
                    total[key] = weights[key].clone()
    for key in total:
        if total[key].is_floating_point() or total[key].is_complex():
            total[key] /= keep
    output = model_dir/"avg_model.pt"
    temporary = output.with_suffix(".pt.tmp")
    torch.save({"models": [total]}, temporary)
    temporary.replace(output)
    save_json(work/"averaging_report.json", {"selection": "lowest validation SISDR loss among saved checkpoints",
              "count": keep, "epochs": [x[1] for x in selected], "validation_losses": [x[0] for x in selected],
              "path": str(output), "sha256": sha256(output), "notice": "Public run.sh contains stale hardcoded epoch arguments; this uses recorded validation scores."})
    print("Averaged model:", output, flush=True)


def build_model(settings):
    import copy
    import torch
    work = Path(settings["work_root"])
    sys.path[:0] = [str(work/"runtime/wesep-real-tse"), str(work/"runtime/wespeaker")]
    from wesep.models import get_model
    cfg = config(settings)
    args = copy.deepcopy(cfg["model_args"]["tse_model"])
    args["speaker"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
    model = get_model(cfg["model"]["tse_model"])(args)
    checkpoint = work/"exp/emb/models/avg_model.pt"
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["models"][0]
    model.load_state_dict(weights, strict=True)
    return model.to("cuda:0").eval()


def audio(path):
    import numpy as np
    import soundfile as sf
    import torch
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000 or not len(waveform):
        raise RuntimeError(f"Invalid evaluation waveform: {path}")
    waveform = np.mean(waveform, axis=1)
    return torch.from_numpy(waveform.copy()).reshape(1, 1, -1).to("cuda:0")


def evaluate_libri_test(settings):
    import torch
    import auraloss
    work = Path(settings["work_root"])
    if not settings["evaluate_libri_test"]:
        return
    model = build_model(settings)
    records = [json.loads(x) for x in (work/"prepared/test/samples.jsonl").read_text().splitlines() if x.strip()]
    registration = json.loads((work/"prepared/test/fixed_enroll.json").read_text())
    loss = auraloss.time.SISDRLoss().to("cuda:0")
    output = work/"evaluation/libri_test_sisdr.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = []
    started = time.perf_counter()
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "speaker", "si_sdr_db"])
        for i, record in enumerate(records, 1):
            mix = audio(record["mix"]["default"][0])
            for speaker in record["spk"]:
                enroll = audio(registration[record["key"]+"::"+speaker][0]["path"])
                target = audio(record["src"][speaker][0])
                with torch.no_grad():
                    extracted = model(mix, [enroll])
                    if isinstance(extracted, (tuple, list)):
                        extracted = extracted[0]
                    score = -float(loss(extracted, target).mean().item())
                if not math.isfinite(score):
                    raise RuntimeError("Nonfinite test SI-SDR")
                totals.append(score)
                writer.writerow([record["key"], speaker, score])
            if i%100 == 0:
                print(f"LibriMix test {i}/{len(records)}", flush=True)
    save_json(work/"evaluation/libri_test_summary.json", {"mixtures": len(records), "target_examples": len(totals),
              "mean_si_sdr_db": sum(totals)/len(totals), "elapsed_hours": (time.perf_counter()-started)/3600,
              "scope": "Full-utterance synthetic test; this is not one of the four REAL-T metrics."})


def evaluate(settings):
    from evaluation import evaluate_real_t
    started = time.perf_counter()
    evaluate_libri_test(settings)
    evaluate_real_t(settings)
    save_json(Path(settings["work_root"])/"evaluation_complete.json", {"status": "complete",
              "libri_test_enabled": settings["evaluate_libri_test"], "real_t_dev_enabled": settings["evaluate_real_t"],
              "elapsed_hours": (time.perf_counter()-started)/3600})


def deliver(settings):
    work = Path(settings["work_root"])
    destination = work/"delivery"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work/"exp/emb/models/avg_model.pt", destination/"avg_model.pt")
    shutil.copy2(work/"exp/emb/config.yaml", destination/"config.yaml")
    shutil.copy2(PACKAGE/"infer_emb.py", destination/"infer_emb.py")
    shutil.copytree(work/"runtime/wesep-real-tse/wesep", destination/"wesep", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(work/"runtime/wespeaker/wespeaker", destination/"wespeaker", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(work/"runtime/wespeaker/LICENSE", destination/"WESPEAKER_LICENSE.txt")
    for source, target in [(work/"prepared/data_report.json", "data_report.json"), (work/"time_estimate.json", "time_estimate.json"),
                           (work/"averaging_report.json", "averaging_report.json"), (work/"runtime/runtime_manifest.json", "runtime_manifest.json"),
                           (work/"training_complete.json", "training_complete.json")]:
        shutil.copy2(source, destination/target)
    for name in ("live_time_estimate.json", "evaluation_complete.json", "environment_selection.json"):
        if (work/name).exists():
            shutil.copy2(work/name, destination/name)
    for source in (work/"evaluation").rglob("*_summary.txt"):
        target = destination/"metrics"/source.relative_to(work/"evaluation")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (work/"evaluation").rglob("*summary.json"):
        target = destination/"metrics"/source.relative_to(work/"evaluation")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    save_json(destination/"delivery_manifest.json", {"files": [{"path": p.relative_to(destination).as_posix(), "sha256": sha256(p)}
              for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "delivery_manifest.json"]})
    archive = shutil.make_archive(str(work/"EMB_Training_Results"), "zip", root_dir=destination)
    print("Results:", archive, flush=True)


if __name__ == "__main__":
    if platform.system() != "Linux":
        raise RuntimeError("Run cloud stages on AutoDL Linux only; local Windows training is disabled.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "prepare_eval", "benchmark", "train", "average", "evaluate", "deliver"], required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    import fcntl
    work = Path(settings["work_root"])
    work.mkdir(parents=True, exist_ok=True)
    with open(work/".pipeline.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("同一工作目录已有云端阶段运行，请勿并行启动两份 Notebook。") from exc
        if args.stage == "prepare":
            from data_prepare import prepare
            prepare(settings)
        elif args.stage == "prepare_eval":
            from evaluation import prepare_eval
            prepare_eval(settings)
        else:
            globals()[args.stage](settings)
