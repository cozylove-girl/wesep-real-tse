"""Small cloud-side range probes; no model imports or full dataset downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

HF_REV = "2dd4206a992cd7a475e1ce2e82b708b014319152"
ARCHIVE_BYTES = {"train-clean-100": 6387309499, "dev-clean": 337926286, "test-clean": 346663984}
BASES = {
    "huggingface": f"https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/{HF_REV}",
    "hf-mirror": f"https://hf-mirror.com/datasets/k2-fsa/LibriSpeech/resolve/{HF_REV}",
    "cn": "https://openslr.magicdatatech.com/resources/12",
    "eu-elda": "https://openslr.elda.org/resources/12",
    "eu-trmal": "https://openslr.trmal.net/resources/12",
    "openslr": "https://www.openslr.org/resources/12",
}


def probe_source(name, subset, *, budget_seconds=12, sample_bytes=1024*1024):
    import requests
    url = BASES[name]+"/"+subset+".tar.gz"
    expected = ARCHIVE_BYTES[subset]
    start = time.monotonic()
    transfer_start = None
    received = 0
    try:
        with requests.Session() as session:
            # AutoDL academic proxy supports HF/GitHub; avoid routing other
            # mirrors through this proxy. Never store signed redirect URLs.
            session.trust_env = name == "huggingface"
            with session.get(url, headers={"Range": f"bytes=0-{sample_bytes-1}"},
                             stream=True, timeout=(15, 8)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes 0-{sample_bytes-1}/{expected}":
                    raise RuntimeError("Source does not honor the exact byte range/official archive size")
                transfer_start = time.monotonic()
                for chunk in response.iter_content(64*1024):
                    received += len(chunk)
                    if received > sample_bytes:
                        raise RuntimeError("Range response exceeded the sample limit")
                    if time.monotonic()-transfer_start >= budget_seconds:
                        break
                if received < sample_bytes and time.monotonic()-transfer_start < budget_seconds:
                    raise RuntimeError("Range sample ended before its declared size")
        elapsed = max(time.monotonic()-start, 0.001)
        transfer_seconds = max(time.monotonic()-transfer_start, 0.001)
        return {"source": name, "url": url, "bytes": received, "seconds": elapsed,
                "connect_seconds": max(0, transfer_start-start), "transfer_seconds": transfer_seconds,
                "mb_per_second": received/transfer_seconds/1e6, "status": "ok"}
    except Exception as error:
        # Only an error type is logged; requests exceptions can contain
        # temporary signed URLs or proxy credentials.
        status = type(error).__name__
        if getattr(error, "response", None) is not None:
            status += f" HTTP {error.response.status_code}"
        return {"source": name, "url": url, "bytes": received, "seconds": time.monotonic()-start,
                "mb_per_second": 0, "status": status}


def select_archive_source(subset, archive_dir):
    requested = os.environ.get("LIBRISPEECH_SOURCE", "auto")
    minimum = float(os.environ.get("LIBRISPEECH_MIN_MBPS", "0"))
    if minimum < 0:
        raise ValueError("LIBRISPEECH_MIN_MBPS must be nonnegative")
    if requested != "auto" and requested not in BASES:
        raise ValueError(f"Unknown LibriSpeech source {requested}; choose auto or {list(BASES)}")
    candidates = list(BASES) if requested == "auto" else [requested]
    print(f"SOURCE_PROBE {subset}: up to 1 MiB/source; minimum {minimum:.2f} MB/s", flush=True)
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        reports = list(pool.map(lambda name: probe_source(name, subset), candidates))
    for report in reports:
        print(f"  {report['source']}: {report['mb_per_second']:.2f} MB/s ({report['status']})", flush=True)
    usable = [r for r in reports if r["status"] == "ok" and r["bytes"] > 0]
    selected = max(usable, key=lambda r: r["mb_per_second"]) if usable else None
    destination = Path(archive_dir)/(subset+".source_probe.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"subset": subset, "hf_revision": HF_REV, "requested": requested,
        "minimum_mb_per_second": minimum, "reports": reports, "selected": selected}, indent=2), encoding="utf-8")
    if selected is None or selected["mb_per_second"] < minimum:
        raise RuntimeError("可用数据源测速过慢，已停止完整下载；查看 *.source_probe.json。可开启学术加速后重试，或上传已有 Libri2Mix 并设置 EXISTING_DATA_ROOT。")
    print(f"SELECT_SOURCE {subset}: {selected['source']} ({selected['mb_per_second']:.2f} MB/s sample; not a full-download guarantee)", flush=True)
    return selected


def source_session(name):
    import requests
    session = requests.Session()
    session.trust_env = name == "huggingface"
    return session
