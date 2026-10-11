"""Prepare only official Libri2Mix train-100/dev/test at 16 kHz, min, mix_clean.

Pairings and gains come from JorisCos/LibriMix. Noise is not read because it is
not part of mix_clean; the published source gains are retained unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if os.name == "nt":
    raise RuntimeError("Prepare this dataset on AutoDL Linux, not on the local Windows machine.")
ROOT = Path(os.environ["EMB_DATA_ROOT"]).resolve()
if os.name == "nt":
    sys.path.insert(0, str(ROOT / "_download" / "python_libs"))
import numpy as np
import requests
import soundfile as sf
from download_sources import ARCHIVE_BYTES, select_archive_source, source_session

ARCHIVES = ROOT / "_download" / "archives" / "LibriSpeech"
SOURCES = ROOT / "sources" / "LibriSpeech"
OUT = ROOT / "simulated" / "Libri2Mix"
METADATA = OUT / "official_metadata"
STATE = ROOT / "_download" / "Libri2Mix-100.json"
SUBSETS = [("train-clean-100", "train-100"), ("dev-clean", "dev"), ("test-clean", "test")]


def save_state(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def digest_file(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_text(url):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(20, 60))
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def cleanup_archive_parts(subset, destination):
    """Remove only generated fragments AFTER the caller verifies full MD5."""
    base = ARCHIVES.resolve()
    data_root = ROOT.resolve()
    destination = Path(destination)
    if (not base.is_relative_to(data_root) or destination.is_symlink()
            or not destination.resolve().is_relative_to(base)
            or not destination.is_file() or destination.stat().st_size != ARCHIVE_BYTES[subset]):
        raise RuntimeError("Unsafe/unverified archive cleanup target")
    parts = ARCHIVES/(subset+".parts")
    if parts.is_symlink():
        raise RuntimeError("Archive fragment directory must not be a symlink")
    candidates = [destination.with_suffix(".gz.part"), destination.with_suffix(destination.suffix+".part")]
    if parts.is_dir():
        candidates.extend(p for p in parts.iterdir() if re.fullmatch(r"\d{12}-\d{12}\.part", p.name)
                          or p.name == "plan.json")
    removed = 0
    for item in candidates:
        if item.is_symlink():
            raise RuntimeError("Archive fragment must not be a symlink")
        if item.is_file():
            if not item.resolve().is_relative_to(base):
                raise RuntimeError("Archive fragment path escaped managed directory")
            removed += item.stat().st_size
            item.unlink()
    if parts.is_dir() and not any(parts.iterdir()):
        parts.rmdir()
    if removed:
        print(f"CLEANUP {subset}: removed {removed/1e9:.3f} GB verified archive fragments", flush=True)


def download_archive_sequential(subset, expected_md5):
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVES / (subset + ".tar.gz")
    if destination.exists():
        print(f"CHECKSUM {destination.name}", flush=True)
        if digest_file(destination) == expected_md5:
            cleanup_archive_parts(subset, destination)
            return destination
        raise RuntimeError(f"Existing archive failed MD5: {destination}")
    part = destination.with_suffix(destination.suffix + ".part")
    urls = [f"https://www.openslr.org/resources/12/{destination.name}",
            f"https://us.openslr.org/resources/12/{destination.name}"]
    for attempt in range(8):
        offset = part.stat().st_size if part.exists() else 0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            url = urls[attempt % len(urls)]
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected HTTP resume offset")
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                expected_bytes = offset + int(response.headers.get("Content-Length", "0"))
                last = time.monotonic()
                with part.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f"DOWNLOAD {destination.name}: {offset/1e9:.3f}/{expected_bytes/1e9:.3f} GB", flush=True)
                            last = time.monotonic()
                if expected_bytes and offset != expected_bytes:
                    raise RuntimeError("Archive response ended before Content-Length")
            print(f"CHECKSUM {destination.name}", flush=True)
            if digest_file(part) != expected_md5:
                raise RuntimeError("Downloaded archive failed official MD5")
            part.replace(destination)
            cleanup_archive_parts(subset, destination)
            print(f"DOWNLOADED {destination.name}: {destination.stat().st_size/1e9:.3f} GB", flush=True)
            return destination
        except Exception as error:
            print(f"RETRY {destination.name} attempt={attempt+1}: {type(error).__name__}: {error}", flush=True)
            if attempt == 7 or "MD5" in str(error):
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def download_archive(subset, expected_md5):
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVES / (subset + ".tar.gz")
    if destination.exists():
        if digest_file(destination) != expected_md5:
            raise RuntimeError(f"Existing archive failed MD5: {destination}")
        cleanup_archive_parts(subset, destination)
        return destination
    selected = select_archive_source(subset, ARCHIVES)
    url = selected["url"]
    prefix = destination.with_suffix(".gz.part")
    parts = ARCHIVES / (subset + ".parts")
    parts.mkdir(exist_ok=True)
    plan_file = parts / "plan.json"
    total = ARCHIVE_BYTES[subset]
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        prefix_bytes = plan["prefix_bytes"]
        if plan["total_bytes"] != total:
            raise RuntimeError("Archive length changed since multipart download began")
        if plan["md5"] != expected_md5:
            raise RuntimeError("Existing archive plan uses different official MD5")
    else:
        prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
        plan = {"total_bytes": total, "prefix_bytes": prefix_bytes, "md5": expected_md5}
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
    span = 64 * 1024 * 1024
    abort = threading.Event()
    intervals = [(start, min(start + span, total) - 1) for start in range(prefix_bytes, total, span)]
    def part_path(start, end):
        return parts / f"{start:012d}-{end:012d}.part"
    def get_part(interval):
        start, end = interval
        target = part_path(start, end)
        size = end - start + 1
        if target.exists() and target.stat().st_size == size:
            return
        for attempt in range(6):
            if abort.is_set():
                raise RuntimeError("Archive download stopped because of slow transfer")
            offset = target.stat().st_size if target.exists() else 0
            actual_start = start + offset
            try:
                with source_session(selected["source"]) as session, session.get(url, headers={"Range": f"bytes={actual_start}-{end}"}, stream=True, timeout=(10, 15)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {actual_start}-{end}/{total}":
                        raise RuntimeError("Server did not honor the requested byte range")
                    with target.open("ab") as stream:
                        for chunk in response.iter_content(64 * 1024):
                            if abort.is_set():
                                raise RuntimeError("Archive download stopped because of slow transfer")
                            if chunk:
                                stream.write(chunk)
                if target.stat().st_size != size:
                    raise RuntimeError("Incomplete archive segment")
                return
            except Exception:
                if abort.is_set():
                    raise
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 20))
    print(f"MULTIPART {destination.name}: {prefix_bytes/1e9:.3f}/{total/1e9:.3f} GB already available", flush=True)
    started = time.monotonic()
    initial_bytes = prefix_bytes+sum(p.stat().st_size for p in parts.glob("*.part"))
    with ThreadPoolExecutor(max_workers=4 if subset == "train-clean-100" else 2) as executor:
        futures = [executor.submit(get_part, interval) for interval in intervals]
        while not all(f.done() for f in futures):
            time.sleep(20)
            available = prefix_bytes + sum(p.stat().st_size for p in parts.glob("*.part"))
            elapsed = max(time.monotonic()-started, 0.001)
            speed = max(0, available-initial_bytes)/elapsed/1e6
            print(f"DOWNLOAD {destination.name}: {available/1e9:.3f}/{total/1e9:.3f} GB; {speed:.2f} MB/s", flush=True)
            minimum = float(os.environ.get("LIBRISPEECH_MIN_MBPS", "0"))
            if minimum > 0 and elapsed >= 120 and speed < minimum:
                abort.set()
                for future in futures:
                    future.cancel()
                raise RuntimeError(f"{destination.name} 持续下载速度 {speed:.2f} MB/s 低于 {minimum:.2f} MB/s，保留分片并停止。开启加速/换源后重试。")
        for future in futures:
            future.result()
    print(f"ASSEMBLE_AND_CHECKSUM {destination.name}", flush=True)
    temporary = destination.with_suffix(".gz.assembling")
    digest = hashlib.md5()
    with temporary.open("wb") as output:
        if prefix_bytes:
            with prefix.open("rb") as original:
                remaining = prefix_bytes
                while remaining:
                    chunk = original.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("Previously downloaded archive prefix is truncated")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
        for start, end in intervals:
            with part_path(start, end).open("rb") as piece:
                for chunk in iter(lambda: piece.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
    if temporary.stat().st_size != total or digest.hexdigest() != expected_md5:
        raise RuntimeError("Assembled archive failed official size/MD5 validation")
    temporary.replace(destination)
    cleanup_archive_parts(subset, destination)
    print(f"DOWNLOADED {destination.name}: {total/1e9:.3f} GB, official MD5 verified", flush=True)
    return destination


def extract_archive(subset, archive_path, expected_md5):
    marker = SOURCES / ("." + subset + ".extracted.json")
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("archive_md5") == expected_md5 and (SOURCES / subset).is_dir():
            print(f"EXTRACTED_ALREADY {subset}", flush=True)
            return
    extraction_root = ROOT / "sources"
    extraction_root.mkdir(parents=True, exist_ok=True)
    source_root = SOURCES.resolve()
    count = 0
    last = time.monotonic()
    print(f"EXTRACT {subset}", flush=True)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            destination = (extraction_root / member.name).resolve()
            if destination != source_root and source_root not in destination.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Missing archive member: {member.name}")
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if destination.stat().st_size != member.size:
                    raise RuntimeError(f"Bad extracted size: {member.name}")
                count += 1
            else:
                raise RuntimeError(f"Unsupported archive member: {member.name}")
            if time.monotonic() - last > 20:
                print(f"EXTRACT {subset}: {count} files", flush=True)
                last = time.monotonic()
    marker.write_text(json.dumps({"archive_md5": expected_md5, "files": count}), encoding="utf-8")
    print(f"EXTRACT_COMPLETE {subset}: {count} files", flush=True)


def valid_wave(path, frames):
    if not path.is_file():
        return False
    try:
        info = sf.info(path)
        return info.samplerate == 16000 and info.channels == 1 and info.frames == frames and info.subtype == "PCM_16"
    except Exception:
        return False


def write_wave(path, samples):
    if valid_wave(path, len(samples)):
        return
    temporary = path.with_suffix(".wav.part")
    sf.write(temporary, samples, 16000, format="WAV", subtype="PCM_16")
    temporary.replace(path)


def make_pair(row, output_dir):
    audio = []
    for i in (1, 2):
        path = SOURCES / row[f"source_{i}_path"]
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise RuntimeError(f"Unexpected source format: {path}")
        audio.append((path, info.frames, float(row[f"source_{i}_gain"])))
    frames = min(a[1] for a in audio)
    sources = [sf.read(path, dtype="float32", frames=frames)[0] * gain for path, _, gain in audio]
    mixture = np.zeros_like(sources[0])
    for source in sources:
        mixture += source
    identifier = row["mixture_ID"]
    paths = [output_dir / directory / (identifier + ".wav") for directory in ("mix_clean", "s1", "s2")]
    for path, signal in zip(paths, (mixture, *sources)):
        write_wave(path, signal)
    snrs = []
    for source in sources:
        noise = mixture - source
        snrs.append(float(10 * np.log10(np.mean(source ** 2) / (np.mean(noise ** 2) + 1e-10) + 1e-10)))
    return [identifier, *(p.resolve().as_posix() for p in paths), frames], [identifier, *snrs]


def generate_subset(source_name, subset):
    output_dir = OUT / "wav16k" / "min" / subset
    for name in ("mix_clean", "s1", "s2"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    with (METADATA / f"libri2mix_{source_name}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata_dir = OUT / "wav16k" / "min" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = metadata_dir / f"mixture_{subset}_mix_clean.csv"
    metrics_path = metadata_dir / f"metrics_{subset}_mix_clean.csv"
    mixture_tmp = mixture_path.with_suffix(".csv.part")
    metrics_tmp = metrics_path.with_suffix(".csv.part")
    print(f"GENERATE {subset}: {len(rows)} official pairs", flush=True)
    last = time.monotonic()
    with mixture_tmp.open("w", newline="", encoding="utf-8") as mixes, metrics_tmp.open("w", newline="", encoding="utf-8") as metrics:
        mix_writer, metric_writer = csv.writer(mixes), csv.writer(metrics)
        mix_writer.writerow(["mixture_ID", "mixture_path", "source_1_path", "source_2_path", "length"])
        metric_writer.writerow(["mixture_ID", "source_1_SNR", "source_2_SNR"])
        with ThreadPoolExecutor(max_workers=8) as executor:
            for count, (mix_row, metric_row) in enumerate(executor.map(lambda row: make_pair(row, output_dir), rows), 1):
                mix_writer.writerow(mix_row)
                metric_writer.writerow(metric_row)
                if time.monotonic() - last > 20:
                    print(f"GENERATE {subset}: {count}/{len(rows)} pairs", flush=True)
                    last = time.monotonic()
    mixture_tmp.replace(mixture_path)
    metrics_tmp.replace(metrics_path)
    counts = {name: len(list((output_dir / name).glob("*.wav"))) for name in ("mix_clean", "s1", "s2")}
    if any(count != len(rows) for count in counts.values()):
        raise RuntimeError(f"Unexpected audio counts for {subset}: {counts}")
    for row in (rows[0], rows[len(rows)//2], rows[-1]):
        identifier = row["mixture_ID"] + ".wav"
        signals = [sf.read(output_dir / name / identifier, dtype="float32")[0] for name in ("mix_clean", "s1", "s2")]
        error = float(np.max(np.abs(signals[0] - signals[1] - signals[2])))
        if error > 3 / 32768:
            raise RuntimeError(f"Mixture/source consistency failed: {subset}/{identifier}: {error}")
    size = sum(p.stat().st_size for p in output_dir.rglob("*.wav"))
    print(f"GENERATED {subset}: {len(rows)} pairs, {size/1e9:.3f} GB", flush=True)
    return {"pairs": len(rows), "wav_counts": counts, "bytes": size}


def source_subset_complete(subset, expected_md5):
    marker = SOURCES / ("." + subset + ".extracted.json")
    metadata = METADATA / f"libri2mix_{subset}.csv"
    if not marker.is_file() or not metadata.is_file():
        return False
    if json.loads(marker.read_text(encoding="utf-8")).get("archive_md5") != expected_md5:
        return False
    paths = set()
    with metadata.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            paths.update((row["source_1_path"], row["source_2_path"]))
    source_root = SOURCES.resolve()
    for relative in paths:
        path = (SOURCES / relative).resolve()
        if source_root not in path.parents or not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def prepare_source_archives(state):
    """Report failures as they happen; retain and extract successful corpora."""
    errors = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for name, _ in SUBSETS:
            expected = state["source_archives"][name]["md5"]
            if source_subset_complete(name, expected):
                print(f"REUSE_EXTRACTED_SOURCES {name}; compressed archive not required", flush=True)
                state["source_archives"][name]["status"] = "extracted"
            else:
                futures[executor.submit(download_archive, name, expected)] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                archive = future.result()
                extract_archive(name, archive, state["source_archives"][name]["md5"])
                state["source_archives"][name]["status"] = "extracted"
            except Exception as error:
                # Keep signed download URLs and proxy credentials out of logs.
                reason = str(error)
                reason = type(error).__name__ if re.search(r"https?://", reason) else reason[:600]
                errors[name] = reason
                state["source_archives"][name].update({"status": "failed", "error": reason})
                print(f"SOURCE_FAILED {name}: {reason}; other downloads continue and completed files are retained", flush=True)
            save_state(state)
    if errors:
        state["status"] = "partial_sources"
        save_state(state)
        raise RuntimeError("部分原始语料未完成；已完成的下载/解压会复用，未完成分片会续传："+json.dumps(errors, ensure_ascii=False))


def main(generate_only):
    METADATA.mkdir(parents=True, exist_ok=True)
    if generate_only:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        # Verified with git ls-remote on 2026-10-09; fixed for reproducibility.
        revision = "b898ca375c03e4039cdb3dbe3dd485aad58ec7d1"
        base = f"https://raw.githubusercontent.com/JorisCos/LibriMix/{revision}"
        state = {"status": "downloading_sources", "source_repository": "https://github.com/JorisCos/LibriMix",
                 "metadata_revision": revision, "sample_rate": 16000, "mode": "min", "mixture_type": "mix_clean",
                 "subsets": {}, "source_archives": {}}
        save_state(state)
        for source_name, _ in SUBSETS:
            for suffix in ("", "_info"):
                name = f"libri2mix_{source_name}{suffix}.csv"
                if not (METADATA / name).is_file():
                    (METADATA / name).write_text(get_text(base + "/metadata/Libri2Mix/" + name), encoding="utf-8")
        if not (METADATA / "LibriMix_LICENSE.txt").is_file():
            (METADATA / "LibriMix_LICENSE.txt").write_text(get_text(base + "/LICENSE"), encoding="utf-8")
        checksum_path = ARCHIVES.parent / "LibriSpeech_md5sum.txt"
        checksum_text = checksum_path.read_text(encoding="utf-8") if checksum_path.exists() else get_text("https://www.openslr.org/resources/12/md5sum.txt")
        (ARCHIVES.parent / "LibriSpeech_md5sum.txt").parent.mkdir(parents=True, exist_ok=True)
        (ARCHIVES.parent / "LibriSpeech_md5sum.txt").write_text(checksum_text, encoding="utf-8")
        checksums = {line.split()[-1].lstrip("*"): line.split()[0] for line in checksum_text.splitlines() if len(line.split()) == 2}
        for source_name, _ in SUBSETS:
            expected = checksums[source_name + ".tar.gz"]
            state["source_archives"][source_name] = {"md5": expected, "source": "https://www.openslr.org/resources/12/" + source_name + ".tar.gz"}
        save_state(state)
        prepare_source_archives(state)
        state["status"] = "generating"
        save_state(state)
    state["status"] = "generating"
    save_state(state)
    for source_name, subset in SUBSETS:
        state["subsets"][subset] = generate_subset(source_name, subset)
        save_state(state)
    state["status"] = "complete"
    state["verification"] = "Official archive MD5; exact published pairs/gains; audio formats/counts; sampled mixture equals source1+source2 within PCM16 quantization tolerance."
    state["destination"] = str(OUT)
    save_state(state)
    print("LIBRI2MIX_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true", help="Rebuild audio/CSV paths from already downloaded sources and metadata.")
    parser.add_argument("--check-sources", action="store_true", help="Check that extracted sources can replace the original archives, without downloading or regenerating.")
    args = parser.parse_args()
    if args.check_sources:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        for name, _ in SUBSETS:
            complete = source_subset_complete(name, state["source_archives"][name]["md5"])
            print(f"{name}: extracted sources complete={complete}", flush=True)
            if not complete:
                raise SystemExit(1)
        raise SystemExit(0)
    try:
        main(args.generate_only)
    except Exception as error:
        if STATE.exists():
            state = json.loads(STATE.read_text(encoding="utf-8"))
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            save_state(state)
        raise
