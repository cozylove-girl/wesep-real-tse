"""Prepare WeSep raw JSONL and independent registration audio; no model execution."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cloud_common import PACKAGE, REPO, download, save_json, sha256


def prepare_lists(librimix_root, output, seed=42):
    """One record per mixture, two targets; use different utterances for enrollment."""
    root, output = Path(librimix_root).resolve(), Path(output).resolve()
    expected = {"train-100": 13900, "dev": 3000, "test": 3000}
    report = {}
    for split, count in expected.items():
        mix_dir = root/"wav16k/min"/split/"mix_clean"
        paths = sorted(mix_dir.glob("*.wav"))
        if len(paths) != count:
            raise RuntimeError(f"{split}: expected {count} mixtures, found {len(paths)} under {mix_dir}")
        records, pool = [], {}
        for mixture in paths:
            utterances = mixture.stem.split("_")
            if len(utterances) != 2:
                raise ValueError(f"Invalid mixture name: {mixture.name}")
            speakers = [uid.split("-")[0] for uid in utterances]
            if speakers[0] == speakers[1]:
                raise ValueError("Libri2Mix must contain different speakers")
            sources = [mixture.parent.parent/name/mixture.name for name in ("s1", "s2")]
            if not all(p.is_file() for p in sources):
                raise FileNotFoundError(f"Missing clean target: {mixture}")
            records.append({"key": mixture.stem, "spk": speakers, "mix": {"default": [str(mixture)]},
                            "src": {spk: [str(src)] for spk, src in zip(speakers, sources)}})
            for speaker, uid, source in zip(speakers, utterances, sources):
                pool.setdefault(speaker, {}).setdefault(uid, {"utt_id": uid, "path": str(source)})
        resources = {speaker: list(values.values()) for speaker, values in pool.items()}
        fixed = {}
        for record in records:
            current = set(record["key"].split("_"))
            for speaker in record["spk"]:
                choices = [item for item in resources[speaker] if item["utt_id"] not in current]
                if not choices:
                    raise RuntimeError(f"No independent registration: {record['key']}::{speaker}")
                key = record["key"]+"::"+speaker
                index = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)%len(choices)
                fixed[key] = [choices[index]]
        destination = output/split
        destination.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records)
        (destination/"samples.jsonl").write_text(text, encoding="utf-8")
        (destination/"raw.list").write_text(text, encoding="utf-8")
        save_json(destination/"speech.json", resources)
        save_json(destination/"fixed_enroll.json", fixed)
        report[split] = {"mixture_records": len(records), "target_examples": 2*len(records),
                         "speakers": len(resources), "independent_registration_verified": True}
    # Standard LibriSpeech train-clean-100 / dev-clean / test-clean have disjoint speaker IDs.
    sets = {split: set(json.loads((output/split/"speech.json").read_text())) for split in expected}
    for first, second in [("train-100", "dev"), ("train-100", "test"), ("dev", "test")]:
        if sets[first]&sets[second]:
            raise RuntimeError(f"Speaker leakage between {first} and {second}")
    return report


def write_cues(directory, split):
    import yaml
    directory = Path(directory)
    training = split == "train-100"
    filename = "speech.json" if training else "fixed_enroll.json"
    cue = {"cues": {"audio": {"type": "raw", "guaranteed": True, "required": True, "scope": "speaker",
           "policy": {"type": "random" if training else "fixed", "key": "spk_id" if training else "mix_spk_id",
                      "resource": (directory/filename).as_posix()}}}}
    text = yaml.safe_dump(cue, sort_keys=False)
    (directory/"cues.yaml").write_text(text, encoding="utf-8")


def stage_runtime(work):
    """Preserve the fork's upstream source; modify only a runtime copy."""
    work = Path(work)
    runtime = work/"runtime/wesep-real-tse"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO/"wesep", runtime/"wesep", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(PACKAGE/"assets/wespeaker_minimal", work/"runtime/wespeaker", dirs_exist_ok=True)
    source_manifest = json.loads((PACKAGE/"assets/wespeaker_sources.json").read_text())
    for source in source_manifest["files"]:
        if sha256(PACKAGE/"assets"/source["path"]) != source["sha256"]:
            raise RuntimeError("Pinned ECAPA dependency source was modified")
    patches = []
    public_api = runtime/"wesep/__init__.py"
    api_text = public_api.read_text(encoding="utf-8")
    if ('from wesep.cli.extractor import load_model  # noqa' not in api_text
            or 'from wesep.cli.extractor import load_model_local  # noqa' not in api_text):
        raise RuntimeError("Unexpected WeSep public API source")
    public_api.write_text('''"""Public inference API is imported only when explicitly requested."""
__all__ = ["load_model", "load_model_local"]

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(".cli.extractor", __name__), name)
    globals()[name] = value
    return value
''', encoding="utf-8")
    patches.append("Lazily export inference APIs; importing training does not load optional Silero VAD")
    speaker = runtime/"wesep/modules/speaker/encoder.py"
    text = speaker.read_text(encoding="utf-8")
    text = text.replace("pretrained_model = torch.load(pretrained)",
                        'pretrained_model = torch.load(pretrained, map_location="cpu", weights_only=False)')
    old = '''            for key in state.keys():
                if key in pretrained_model:
                    state[key] = pretrained_model[key]
                else:
                    print(f"not {key} loaded")

            self.spk_model.load_state_dict(state)'''
    new = '''            required = set(dict(self.spk_model.named_parameters()))
            missing = required - set(pretrained_model)
            if missing:
                raise RuntimeError(f"Pretrained ECAPA learned parameters missing: {sorted(missing)}")
            missing_buffers = set(state) - set(pretrained_model) - required
            if any(not key.endswith("num_batches_tracked") for key in missing_buffers):
                raise RuntimeError(f"Pretrained ECAPA buffers missing: {sorted(missing_buffers)}")
            for key in state:
                if key in pretrained_model:
                    state[key] = pretrained_model[key]
            self.spk_model.load_state_dict(state, strict=True)'''
    if text.count(old) != 1:
        raise RuntimeError("Unexpected SpeakerEncoder source; cannot apply verified patch")
    speaker.write_text(text.replace(old, new), encoding="utf-8")
    patches.append("Strictly require all ECAPA learned parameters; ignore checkpoint-only classification head")

    cues = runtime/"wesep/dataset/processor_speaker.py"
    text = cues.read_text(encoding="utf-8")
    old = "            items = spk_resource[lookup_key]"
    new = '''            current_utterances = set(sample["key"].split("_"))
            items = [item for item in spk_resource[lookup_key]
                     if item.get("utt_id") not in current_utterances]'''
    if text.count(old) != 2:
        raise RuntimeError("Unexpected registration sampler source")
    cues.write_text(text.replace(old, new), encoding="utf-8")
    patches.append("Exclude mixture utterances when sampling registration audio")

    checkpoint = runtime/"wesep/utils/checkpoint.py"
    text = checkpoint.read_text(encoding="utf-8")
    text = "import os\n"+text
    text = text.replace('map_location="cpu",', 'map_location="cpu",\n        weights_only=False,')
    text = text.replace("strict=False", "strict=True")
    old = "        path,\n    )"
    if text.count(old) != 1:
        raise RuntimeError("Unexpected checkpoint writer source")
    checkpoint.write_text(text.replace(old, "        path + '.tmp',\n    )\n    os.replace(path + '.tmp', path)"), encoding="utf-8")
    patches.extend(["Strict resume and PyTorch 2.7 checkpoint compatibility", "Atomic checkpoint writes"])
    # Validate every edited source before starting cloud processes.
    for file in (speaker, cues, checkpoint, public_api):
        ast.parse(file.read_text(encoding="utf-8"))
    save_json(work/"runtime/runtime_manifest.json", {"wesep_commit": "2a540977a348fbaa92e623210505430e2cec608d",
              "wespeaker_commit": source_manifest["revision"], "patches": patches,
              "source_train_sha256": sha256(REPO/"wesep/bin/train.py"),
              "patched_files": {str(p.relative_to(runtime)): sha256(p) for p in (speaker, cues, checkpoint, public_api)}})
    return runtime


def prepare(settings):
    import yaml
    work = Path(settings["work_root"])
    existing = settings.get("existing_data_root", "")
    data = Path(existing).resolve() if existing else work/"data"
    if existing:
        if not data.is_absolute() or not data.exists():
            raise FileNotFoundError(data)
    else:
        metadata = data/"simulated/Libri2Mix/official_metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        for source in (PACKAGE/"assets/librimix_metadata").iterdir():
            shutil.copy2(source, metadata/source.name)
        checksums = data/"_download/archives/LibriSpeech_md5sum.txt"
        checksums.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE/"assets/LibriSpeech_md5sum.txt", checksums)
        completion = data/"_download/Libri2Mix-100.json"
        ready = completion.exists() and json.loads(completion.read_text()).get("status") == "complete"
        ready = ready and all(len(list((data/"simulated/Libri2Mix/wav16k/min"/split/name).glob("*.wav"))) == count
                              for split, count in [("train-100", 13900), ("dev", 3000), ("test", 3000)] for name in ("mix_clean", "s1", "s2"))
        if not ready:
            subprocess.run([sys.executable, str(PACKAGE/"prepare_librimix.py")], check=True)
        else:
            print("复用已完整生成的 Libri2Mix；继续检查注册语音与路径。", flush=True)
    report = prepare_lists(data/"simulated/Libri2Mix", work/"prepared", settings["seed"])
    for split in report:
        write_cues(work/"prepared"/split, split)
    speaker = work/"models/ecapa512/avg_model.pt"
    download(settings["hf_endpoint"].rstrip("/")+"/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/resolve/89426e111abc9ffcc40279baf808e087693c6e9b/avg_model.pt",
             speaker, checksum="c5659521b311ce284751b7a7e8897e9287b7b9131db00ad58f35807b49515f6a", size=38708805)
    runtime = stage_runtime(work)
    cfg = yaml.safe_load((REPO/"examples/audio/librimix/confs/tse_bsrnn_spk.yaml").read_text(encoding="utf-8"))
    cfg.update({"seed": settings["seed"], "exp_dir": str(work/"exp/emb"), "gpus": list(range(len(settings["gpu_ids"].split(",")))),
                "num_epochs": settings["epochs"], "data_type": "raw", "repeat_dataset": True,
                "log_batch_interval": 50, "find_unused_parameters": False})
    cfg["dataloader_args"].update({"batch_size": settings["batch_size"], "num_workers": settings["num_workers"],
                                    "prefetch_factor": 2, "pin_memory": True, "drop_last": True})
    if settings["num_workers"] == 0:
        cfg["dataloader_args"].pop("prefetch_factor", None)
    cfg["dataset_args"]["shuffle_args"]["shuffle_size"] = settings["shuffle_buffer"]
    cfg["model_args"]["tse_model"]["speaker"]["speaker_model"]["speaker_encoder"].update({"pretrained": str(speaker), "freeze": True})
    cfg["model_init"]["tse_model"] = settings.get("initial_tse_checkpoint") or None
    if cfg["model_init"]["tse_model"] and not Path(cfg["model_init"]["tse_model"]).is_file():
        raise FileNotFoundError(cfg["model_init"]["tse_model"])
    for stage, split in (("train", "train-100"), ("val", "dev")):
        cfg[stage+"_data"] = str(work/"prepared"/split/"raw.list")
        cfg[stage+"_samples"] = str(work/"prepared"/split/"samples.jsonl")
        cfg[stage+"_cues"] = str(work/"prepared"/split/"cues.yaml")
    identity = {"config": cfg, "source_train_sha256": sha256(REPO/"wesep/bin/train.py"),
                "runtime_manifest": sha256(work/"runtime/runtime_manifest.json"),
                "requirements_sha256": sha256(PACKAGE/"requirements-cloud.txt"),
                "core_source_sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted((REPO/"wesep").rglob("*.py"))}}
    state_file = work/"experiment_identity.json"
    environment_file = work/"environment_selection.json"
    if environment_file.exists():
        identity["environment"] = json.loads(environment_file.read_text(encoding="utf-8"))
    if state_file.exists() and json.loads(state_file.read_text(encoding="utf-8")) != identity:
        if any((work/"exp/emb/models").glob("checkpoint_*.pt")):
            raise RuntimeError("已有正式训练权重且配置/源码已变化；请选新的 WORK_ROOT，避免混用不同实验。")
        print("正式训练尚未保存 checkpoint，允许调整 batch/预检参数并复用已有数据。", flush=True)
    save_json(state_file, identity)
    config = work/"prepared/train.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report.update({"data_root": str(data), "config": str(config), "architecture": "offline BSRNN + global ECAPA embedding",
                   "separator_start": "compatible checkpoint" if cfg["model_init"]["tse_model"] else "random initialization",
                   "speaker_encoder": "pretrained ECAPA512 frozen parameters", "corpus": "Libri2Mix-100 16k/min/mix_clean"})
    save_json(work/"prepared/data_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if settings.get("remove_source_archives", True) and not existing:
        # Delete only the three verified compressed duplicates after successful data preparation.
        checksums = dict((line.split()[1], line.split()[0]) for line in (PACKAGE/"assets/LibriSpeech_md5sum.txt").read_text().splitlines() if len(line.split()) == 2)
        for name in ("train-clean-100", "dev-clean", "test-clean"):
            archive = data/"_download/archives/LibriSpeech"/(name+".tar.gz")
            marker = data/"sources/LibriSpeech"/("."+name+".extracted.json")
            if archive.exists() and marker.exists() and json.loads(marker.read_text())["archive_md5"] == checksums[name+".tar.gz"]:
                if not archive.resolve().is_relative_to((work/"data").resolve()) or archive.is_symlink():
                    raise ValueError("Archive cleanup target outside work data")
                archive.unlink()
    return report
