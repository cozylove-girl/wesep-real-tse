"""Read-only NVIDIA memory sampling for cloud training; no torch imports."""
import csv
import json
from pathlib import Path
import subprocess
import threading
import time


def parse_gpu_rows(text):
    result = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        try:
            result.append({"index": row[0].strip(), "name": row[1].strip(),
                           "total_mib": float(row[2]), "used_mib": float(row[3]),
                           "utilization_percent": float(row[4])})
        except ValueError:
            continue
    return result


def gpu_snapshot(gpu_ids):
    command = ["nvidia-smi", "--id="+str(gpu_ids),
               "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    rows = parse_gpu_rows(reply.stdout)
    if not rows:
        raise RuntimeError("nvidia-smi did not return supported memory fields")
    return rows


class GPUMemoryMonitor:
    def __init__(self, gpu_ids, report_file, *, interval_seconds=1, emit_seconds=10):
        self.gpu_ids = gpu_ids
        self.report_file = Path(report_file)
        self.interval_seconds = interval_seconds
        self.emit_seconds = emit_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.peaks = {}
        self.samples = 0
        self.last_error = None
        self.last_emit = 0
        self.status = "running"

    def sample(self, *, emit=False):
        try:
            rows = gpu_snapshot(self.gpu_ids)
            self.samples += 1
            for row in rows:
                old = self.peaks.get(row["index"], {})
                row["observed_peak_used_mib"] = max(row["used_mib"], old.get("observed_peak_used_mib", 0))
                self.peaks[row["index"]] = row
            self.last_error = None
        except Exception as error:
            self.last_error = type(error).__name__
        if emit:
            for row in self.peaks.values():
                print(f"VRAM GPU {row['index']}: {row['used_mib']/1024:.2f}/{row['total_mib']/1024:.2f} GiB; "
                      f"observed peak {row['observed_peak_used_mib']/1024:.2f} GiB; GPU {row['utilization_percent']:.0f}%", flush=True)
            if self.last_error:
                print("VRAM_MONITOR unavailable:", self.last_error, flush=True)
            self.save()

    def save(self):
        data = {"status": self.status, "updated_unix_seconds": time.time(), "samples": self.samples,
                "sample_interval_seconds": self.interval_seconds, "devices": list(self.peaks.values()),
                "last_error": self.last_error,
                "notice": "nvidia-smi measures total device use; 1s sampling can miss brief peaks and includes other processes on this GPU."}
        try:
            self.report_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.report_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(self.report_file)
        except OSError as error:
            self.last_error = "monitor_report_"+type(error).__name__
            print("VRAM_MONITOR report unavailable:", self.last_error, flush=True)

    def loop(self):
        while not self.stop_event.wait(self.interval_seconds):
            emit = time.monotonic()-self.last_emit >= self.emit_seconds
            self.sample(emit=emit)
            if emit:
                self.last_emit = time.monotonic()

    def __enter__(self):
        self.sample(emit=True)
        self.last_emit = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=7)
        self.status = "failed" if kind is not None else "finished"
        self.sample(emit=True)
