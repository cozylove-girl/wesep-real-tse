"""Cloud-only evaluation model load checks before costly formal training."""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Evaluation model checks run only on cloud Linux; no local model execution.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    from cloud_common import save_json
    import torch
    import sherpa_onnx
    import onnxruntime as ort
    repo, work = args.repo.resolve(), args.work_root.resolve()
    results = {}
    for release, epoch in [("sherpa-onnx-zipformer-gigaspeech-2023-12-12", 30),
                           ("sherpa-onnx-zipformer-multi-zh-hans-2023-9-2", 20)]:
        folder = repo/"zipformer/pretrained_models"/release
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(folder/f"encoder-epoch-{epoch}-avg-1.onnx"),
            decoder=str(folder/f"decoder-epoch-{epoch}-avg-1.onnx"),
            joiner=str(folder/f"joiner-epoch-{epoch}-avg-1.onnx"),
            tokens=str(folder/"tokens.txt"), num_threads=1,
            sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
        results[release] = "loaded_fp32"
        del recognizer
        gc.collect()
        print("EVAL_MODEL_LOAD passed:", release, flush=True)
    sys.path.insert(0, str(repo/"FireRedASR2S"))
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
    vad = FireRedVad.from_pretrained(str(repo/"FireRedASR2S/pretrained_models/FireRedVAD/VAD"),
                                    FireRedVadConfig(use_gpu=torch.cuda.is_available()))
    results["firered_vad"] = "loaded"
    del vad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("EVAL_MODEL_LOAD passed: FireRedVAD", flush=True)
    paths = [work/"models/wespeaker_evaluation/en/model.onnx", work/"models/wespeaker_evaluation/chs/model.onnx",
             work/"models/DNSMOS/sig_bak_ovr.onnx", work/"models/DNSMOS/model_v8.onnx"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for path in paths:
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if not session.get_inputs() or not session.get_outputs():
            raise RuntimeError("Evaluation ONNX model has no inputs/outputs")
        results[str(path.relative_to(work))] = "loaded_cpu"
        del session
        gc.collect()
        print("EVAL_MODEL_LOAD passed:", path.name, flush=True)
    save_json(work/"evaluation_model_check.json", {"status": "passed", "models": results,
              "scope": "Model construction/load only, no training or metric inference. Full evaluation still required.",
              "torch": torch.__version__, "onnxruntime": ort.__version__})


if __name__ == "__main__":
    main()
