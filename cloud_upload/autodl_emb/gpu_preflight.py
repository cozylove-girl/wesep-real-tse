"""Real CUDA operator checks before downloading the training corpus."""
import argparse
import json
from pathlib import Path
import platform

import torch
import torchaudio


def check_devices():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：核对是否为有卡实例及 NVIDIA 驱动/所装 wheel。")
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        # cuDNN LSTM backward and STFT backward are both used by BSRNN.
        rnn = torch.nn.LSTM(8, 16, batch_first=True).to(device).train()
        signal = torch.randn(1, 4096, device=device, requires_grad=True)
        spectrum = torch.stft(signal, n_fft=512, hop_length=128,
                              window=torch.hann_window(512, device=device), return_complex=True)
        spectrum.abs().square().mean().backward()
        assert signal.grad is not None and torch.isfinite(signal.grad).all()
        frames = torch.randn(1, 32, 8, device=device, requires_grad=True)
        output, _ = rnn(frames)
        output.square().mean().backward()
        assert frames.grad is not None and torch.isfinite(frames.grad).all()
        torch.cuda.synchronize(device)
        devices.append({"visible_index": index, "name": torch.cuda.get_device_name(index),
                        "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                        "stft_backward": True, "cudnn_lstm_backward": True})
        del rnn, signal, spectrum, frames, output
        torch.cuda.empty_cache()
    return {"status": "passed", "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(), "glibc": list(platform.libc_ver()), "devices": devices,
            "notice": "These small operator checks do not establish PS4 full-training VRAM requirements."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = check_devices()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
