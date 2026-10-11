"""Inference for the delivered WeSep EMB model. This script never trains."""
import argparse
import copy
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_dir))
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import yaml
    from wesep.models import get_model
    cfg = yaml.safe_load((args.model_dir/"config.yaml").read_text(encoding="utf-8"))
    model_args = copy.deepcopy(cfg["model_args"]["tse_model"])
    model_args["speaker"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
    model = get_model(cfg["model"]["tse_model"])(model_args)
    checkpoint = torch.load(args.model_dir/"avg_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["models"][0], strict=True)
    model.to(args.device).eval()
    def read(path):
        signal, sr = sf.read(path, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(signal.mean(axis=1).copy()).reshape(1, 1, -1)
        if sr != 16000:
            tensor = torchaudio.functional.resample(tensor, sr, 16000)
        return tensor.to(args.device)
    with torch.no_grad():
        result = model(read(args.mix), [read(args.enroll)])
        if isinstance(result, (tuple, list)):
            result = result[0]
        waveform = result.detach().cpu().reshape(-1).numpy()
    if not np.isfinite(waveform).all():
        raise RuntimeError("Invalid output")
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform/peak*0.9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, 16000, subtype="FLOAT")
    print(args.output)


if __name__ == "__main__":
    main()
