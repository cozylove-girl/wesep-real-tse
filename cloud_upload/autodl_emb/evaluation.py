"""Official REAL-T evaluator setup shared with the reviewed PS4 cloud workflow."""
import csv
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cloud_common import PACKAGE, download, save_json, sha256, runtime_selection, configure_hub_transport
EVAL_REV = "8bde4f147d452de43834b5c0a4efdb85811fd5dd"
EN_REV = "f0c48c298fd835726c27956a5d617bad7115627e"
ZH_REV = "ed25c1ecbd9e8d3c7944df95430da02bd3cdcf7a"
DNS_REV = "9d85678cc47b74531aadba5780b4b699f65a4de7"


def hf_snapshot(repo, revision, target, patterns, repo_type="model", workers=8):
    import fnmatch
    configure_hub_transport()
    from huggingface_hub import HfApi, snapshot_download
    result = snapshot_download(repo_id=repo, revision=revision, repo_type=repo_type,
                               local_dir=str(target), allow_patterns=patterns, max_workers=workers)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    verified = []
    for item in info.siblings:
        if not any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns):
            continue
        path = Path(target) / item.rfilename
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise RuntimeError(f"模型文件缺失/长度不符: {path}")
        actual_sha = sha256(path)
        remote_sha = item.lfs.sha256 if item.lfs else None
        if remote_sha and actual_sha != remote_sha:
            raise RuntimeError(f"模型文件 SHA256 不符: {path}；移走损坏文件后重新下载。")
        verified.append({"path": item.rfilename, "bytes": path.stat().st_size, "sha256": actual_sha})
    save_json(Path(target) / "cloud_source_manifest.json", {"repository": repo, "revision": revision, "files": verified})
    return result

def dataset_download(manifest_file, target, endpoint, workers):
    """Download original files to raw; generated absolute mappings live in a separate view."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import quote
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    files = manifest["files"]
    root = Path(target)
    base = endpoint.rstrip("/") + "/datasets/" + manifest["repository"] + "/resolve/" + manifest["revision"] + "/"
    def fetch(item):
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(item["path"])
        return download(base + quote(item["path"], safe="/"), path,
                        checksum=item.get("sha256"), size=item["bytes"])
    print(f"数据下载/校验 {manifest['repository']}：{len(files)} files", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in files]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)
    return root

def data_view(raw, view, training=True):
    """Symlink audio/annotations and write ONLY cloud mappings; raw download remains intact."""
    raw, view = Path(raw).resolve(), Path(view).resolve()
    view.mkdir(parents=True, exist_ok=True)
    roots = [raw / n for n in ("CHiME6", "AMI", "AliMeeting", "AISHELL-4")] if training else [raw / "REAL-T-dev"]
    for source in roots:
        if not source.is_dir():
            raise FileNotFoundError(source)
        dest = view / source.name
        dest.mkdir(exist_ok=True)
        for child in source.iterdir():
            if child.name.startswith("mapping.") or child.name.startswith(".cache"):
                continue
            link = dest / child.name
            if link.is_symlink():
                if link.resolve() != child.resolve():
                    raise RuntimeError(f"已有链接指向不同数据: {link}")
            elif not link.exists():
                link.symlink_to(child, target_is_directory=child.is_dir())
        audio = sorted(source.rglob("*.wav"))
        mapping = {p.stem: str(p) for p in audio}
        if len(mapping) != len(audio):
            raise ValueError(f"Duplicate utterance IDs: {source}")
        with open(dest / "mapping.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["utterance", "path"])
            writer.writerows(mapping.items())
    return view

def prepare_core_models(settings, repo):
    """Fetch the evaluator's FP32 models without pre.sh GitHub urllib."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote
    repo = Path(repo).resolve()
    manifest = json.loads((PACKAGE/"assets/evaluation_core_models.json").read_text(encoding="utf-8"))
    endpoint = settings["hf_endpoint"].rstrip("/")
    prepared = []
    for model in manifest["models"]:
        destination = (repo/model["destination"]).resolve()
        if not destination.is_relative_to(repo):
            raise ValueError("Evaluation model destination escaped repository")
        destination.mkdir(parents=True, exist_ok=True)
        def fetch(item):
            target = (destination/item["path"]).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Evaluation model filename escaped destination")
            url = endpoint+"/"+model["repository"]+"/resolve/"+model["revision"]+"/"+quote(item["path"], safe="/")
            return download(url, target, checksum=item["sha256"], size=item["bytes"])
        print(f"PREPARE_EVAL_MODEL {model['key']}: fixed FP32/VAD files via {endpoint}", flush=True)
        with ThreadPoolExecutor(max_workers=min(3, settings.get("download_workers", 3))) as pool:
            list(pool.map(fetch, model["files"]))
        prepared.append({**model, "resolved_destination": str(destination)})
        save_json(destination/"cloud_source_manifest.json", model)
    save_json(Path(settings["work_root"])/"evaluation_core_models.json", {"models": prepared,
              "transport": endpoint, "tls_verification": True, "file_size_and_sha256_verified": True})


def disable_pre_script_downloads(env):
    result = dict(env)
    for suffix in ("ZIPFORMER_EN", "ZIPFORMER_ZH", "FIRERED_ASR", "WHISPER", "FIRERED_VAD", "DNSMOS"):
        result["REALT_PREP_DOWNLOAD_"+suffix] = "0"
    return result


def patch_firered_vad_import(repo):
    """VAD is standalone; avoid importing unused ASR/LLM/punctuation APIs."""
    root = Path(repo)/"FireRedASR2S/fireredasr2s/__init__.py"
    text = root.read_text(encoding="utf-8")
    old = '''from fireredasr2s.fireredasr2system import (
    FireRedAsr2System,
    FireRedAsr2SystemConfig
)'''
    lazy = '''# EMB_EVAL_LAZY_ASR_SYSTEM
def __getattr__(name):
    if name not in ("FireRedAsr2System", "FireRedAsr2SystemConfig"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(".fireredasr2system", __name__), name)
    globals()[name] = value
    return value'''
    if old in text:
        text = text.replace(old, lazy)
        ast.parse(text)
        root.write_text(text, encoding="utf-8")
    elif "# EMB_EVAL_LAZY_ASR_SYSTEM" not in text:
        raise RuntimeError("Unexpected FireRedASR2S package initializer")
    return sha256(root)


def evaluation_environment(settings):
    """Prepare evaluation dependencies BEFORE the lengthy training phase."""
    work = Path(settings["work_root"])
    repo = work / "code/REAL-TSE-Challenge"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/REAL-TSE/REAL-TSE-Challenge.git", str(repo)], check=True)
        subprocess.run(["git", "checkout", EVAL_REV], cwd=repo, check=True)
        subprocess.run(["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
                        "submodule", "update", "--init", "--recursive"], cwd=repo, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != EVAL_REV:
        raise RuntimeError("Evaluation repo revision changed")
    # Repair an interrupted submodule initialization too.
    subprocess.run(["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
                    "submodule", "update", "--init", "--recursive"], cwd=repo, check=True)
    # Install only actual evaluation dependencies, keep the training package pins.
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(PACKAGE / "requirements-eval.txt"),
                    "-c", str(PACKAGE / "requirements-cloud.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "wespeakerruntime==1.0.1"], check=True)
    check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    issues = [line for line in check.stdout.splitlines() if line.strip()]
    accepted = "wespeakerruntime 1.0.1 requires onnxruntime, which is not installed."
    if check.returncode and (not issues or any(line.strip().lower() != accepted for line in issues)):
        raise RuntimeError(f"Evaluation dependency conflicts: {check.stdout}\n{check.stderr}")
    if check.returncode:
        print("WeSpeaker dependency metadata asks for CPU onnxruntime; onnxruntime-gpu supplies the same API. CPU package intentionally excluded.", flush=True)
    # Runtime 1.0.1 hardcodes ~/.wespeaker. Patch only our isolated environment
    # to use a dedicated data-disk cache and pin its two ONNX models to HF revisions.
    from wespeakerruntime.hub import Hub
    import inspect
    hub_file = Path(inspect.getfile(Hub))
    hub_text = hub_file.read_text(encoding="utf-8")
    old = 'model_dir = os.path.join(Path.home(), ".wespeaker", lang)'
    new = 'model_dir = os.path.join(os.environ.get("PS4_WESPEAKER_CACHE", str(Path.home() / ".wespeaker")), lang)'
    if old in hub_text:
        hub_file.write_text(hub_text.replace(old, new), encoding="utf-8")
    elif new not in hub_text:
        raise RuntimeError("Unsupported WeSpeaker runtime cache implementation")
    for lang, source_repo, revision, filename in [
        ("en", "Wespeaker/wespeaker-resnet34-LM", EN_REV, "voxceleb_resnet34_LM.onnx"),
        ("chs", "Wespeaker/wespeaker-cnceleb-resnet34-LM", ZH_REV, "cnceleb_resnet34_LM.onnx")]:
        dest = work / "models/wespeaker_evaluation" / lang
        hf_snapshot(source_repo, revision, dest, [filename])
        link = dest / "model.onnx"
        if not link.exists():
            link.symlink_to(dest / filename)
    selected = runtime_selection(settings)
    env = os.environ.copy()
    env.update({"REALT_CONDA_ENV": selected["prefix"], "REALT_PREP_DOWNLOAD_WHISPER": "0",
                "REALT_PREP_DOWNLOAD_FIRERED_ASR": "0", "DNSMOS_MODEL_DIR": str(work / "models/DNSMOS"),
                "REALT_DNSMOS_MODEL_DIR": str(work / "models/DNSMOS"),
                "PS4_WESPEAKER_CACHE": str(work / "models/wespeaker_evaluation"),
                "PATH": str(Path(selected["python"]).parent) + os.pathsep + os.environ["PATH"]})
    # Notebook and conda prefix can coexist. Source conda hook before the upstream env_setup.sh.
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    conda_base = subprocess.check_output([conda, "info", "--base"], text=True).strip()
    hook = Path(conda_base) / "etc/profile.d/conda.sh"
    hf_snapshot("Vyvo-Research/dnsmos", DNS_REV, work / "models/DNSMOS", ["sig_bak_ovr.onnx", "model_v8.onnx"])
    prepare_core_models(settings, repo)
    patch_firered_vad_import(repo)
    env = disable_pre_script_downloads(env)
    subprocess.run(["bash", "-c", 'source "$1"; bash ./pre.sh', "emb-eval", str(hook)], cwd=repo, env=env, check=True)
    subprocess.run([sys.executable, str(PACKAGE/"evaluation_model_check.py"),
                    "--repo", str(repo), "--work-root", str(work)], env=env, check=True)
    return repo, env, hook


def prepare_eval(settings):
    if not settings["evaluate_real_t"]:
        return
    work = Path(settings["work_root"])
    raw = dataset_download(PACKAGE/"assets/real_t_dev_manifest.json", work/"data/real_t_raw",
                           settings["hf_endpoint"], settings["download_workers"])
    data_view(raw, work/"data/real_t_view", training=False)
    repo, _, _ = evaluation_environment(settings)
    save_json(work/"evaluation_environment.json", {"status": "prepared", "repo": str(repo),
                                                  "revision": EVAL_REV, "split": "DEV"})


def evaluate_real_t(settings):
    if not settings["evaluate_real_t"]:
        return
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from pipeline_worker import audio, build_model
    work = Path(settings["work_root"])
    repo, env, hook = evaluation_environment(settings)
    dev = work/"data/real_t_view/REAL-T-dev"
    checkpoint = work/"exp/emb/models/avg_model.pt"
    tag = "emb_avg_"+sha256(checkpoint)[:10]
    output = work/"evaluation/REAL-T/DEV"/tag
    model = build_model(settings)
    with open(dev/"mapping.csv", encoding="utf-8", newline="") as stream:
        mapping = {row["utterance"]: row["path"] for row in csv.DictReader(stream)}
    expected_count = 0
    started = time.perf_counter()
    for meta in sorted((dev/"DEV").glob("*_meta.csv")):
        name = meta.stem.removesuffix("_meta")
        directory = output/name/"wav"
        directory.mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(open(meta, encoding="utf-8", newline="")))
        expected_count += len(rows)
        for i, row in enumerate(rows, 1):
            mix_id, enroll_id = row["mixture_utterance"], row["enrolment_speakers_utterance"]
            target = directory/f"{mix_id}-{enroll_id}.wav"
            expected_frames = sf.info(mapping[mix_id]).frames
            if target.is_file():
                info = sf.info(target)
                if info.samplerate == 16000 and info.channels == 1 and info.frames == expected_frames:
                    continue
            mix, enroll = audio(mapping[mix_id]), audio(mapping[enroll_id])
            with torch.no_grad():
                result = model(mix, [enroll])
                if isinstance(result, (tuple, list)):
                    result = result[0]
                waveform = result.detach().cpu().reshape(-1).numpy()
            if len(waveform) != expected_frames or not np.isfinite(waveform).all():
                raise RuntimeError(f"Invalid TSE output: {target}")
            # Match WeSep Extractor's documented default output peak normalization.
            peak = float(np.max(np.abs(waveform)))
            if peak > 0:
                waveform = waveform/peak*0.9
            temporary = target.with_suffix(".part.wav")
            sf.write(temporary, waveform, 16000, subtype="FLOAT")
            temporary.replace(target)
            if i%50 == 0:
                print(f"REAL-T {name}: {i}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    subprocess.run(["bash", "-c", 'source "$1"; shift; bash ./run_eval.sh "$@"', "emb-eval", str(hook),
                    "--output-dir", str(output), "--test-set", "DEV", "--cuda", settings["gpu_ids"].split(",")[0],
                    "--dataset-root", str(dev)], cwd=repo, env=env, check=True)
    metrics = {"TER": ("TER", "wer_or_cer"), "SpkSim": ("spk_similarity", "speaker_cosine_similarity"),
               "DNSMOS-P808": ("dnsmos", "P808"), "F1": ("TSE_TIMING", "f1")}
    means = {}
    for metric, (suffix, column) in metrics.items():
        table = pd.read_csv(output/"eval_metrics"/f"{tag}_{suffix}.csv")
        if len(table) != expected_count or column not in table or not np.isfinite(table[column]).all():
            raise RuntimeError(f"Incomplete {metric} evaluation; cannot report success")
        if "status" in table and not (table["status"] == "ok").all():
            raise RuntimeError(f"Failed {metric} samples")
        means[metric] = float(table[column].mean())
    save_json(work/"evaluation/real_t_summary.json", {"split": "DEV", "samples": expected_count,
              "metrics": means, "elapsed_hours": (time.perf_counter()-started)/3600,
              "output": str(output), "evaluation_revision": EVAL_REV, "checkpoint_sha256": sha256(checkpoint)})
    print(json.dumps(means, ensure_ascii=False, indent=2), flush=True)
