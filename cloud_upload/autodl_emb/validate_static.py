"""Validate files, data registration rules and timing arithmetic. NO torch/model execution."""
import argparse
import ast
import json
from pathlib import Path
import tempfile

from cloud_common import PACKAGE, REPO, save_json, sha256
from data_prepare import prepare_lists, stage_runtime
from time_estimate import estimate, describe
from cloud_entry import TimedLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librimix-root", type=Path, required=True)
    parser.add_argument("--check-root", type=Path, required=True)
    args = parser.parse_args()
    check_root = args.check_root.resolve()
    workspace = REPO.parents[1].resolve()
    if not check_root.is_relative_to(workspace):
        raise ValueError("Validation scratch must remain inside the workspace")
    check_root.mkdir(parents=True, exist_ok=True)
    for file in PACKAGE.rglob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"))
    report = {"all_python_ast": "passed", "model_executed_locally": False, "training_executed_locally": False,
              "cloud_gpu_executed": False}
    notebook = json.loads((PACKAGE/"EMB_AutoDL_Full_Pipeline.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            if cell["outputs"] or cell["execution_count"] is not None:
                raise AssertionError("Notebook should ship unexecuted")
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    report["notebook_schema"] = "passed"
    report["notebook_cells"] = len(notebook["cells"])

    source_before = {p.relative_to(REPO).as_posix(): sha256(p) for p in (REPO/"wesep").rglob("*.py")}
    # Resolved absolute scope is verified BEFORE TemporaryDirectory performs recursive cleanup.
    with tempfile.TemporaryDirectory(dir=check_root, prefix="emb-static-") as temporary:
        root = Path(temporary).resolve()
        if not root.is_relative_to(check_root) or not root.is_relative_to(workspace):
            raise AssertionError("Unsafe scratch path")
        runtime = stage_runtime(root)
        for file in runtime.rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8"))
        report["runtime_patches_ast"] = "passed"
        data = prepare_lists(args.librimix_root, root/"prepared")
        assert data["train-100"]["mixture_records"] == 13900
        assert data["train-100"]["target_examples"] == 27800
        for split in ("train-100", "dev", "test"):
            fixed = json.loads((root/"prepared"/split/"fixed_enroll.json").read_text())
            for key, items in fixed.items():
                mixture, speaker = key.split("::")
                assert items[0]["utt_id"] not in mixture.split("_")
                assert items[0]["utt_id"].split("-")[0] == speaker
                assert Path(items[0]["path"]).is_file()
        report["data_registration"] = data
        report["no_same_utterance_enrollment"] = True

    source_after = {p.relative_to(REPO).as_posix(): sha256(p) for p in (REPO/"wesep").rglob("*.py")}
    assert source_before == source_after
    report["upstream_source_preserved"] = True
    example = estimate(train_pairs=13900, val_pairs=3000, mixtures_per_batch=15, world_size=1, epochs=150,
                       train_seconds=0.5, val_seconds=0.2)
    assert example["train_steps_per_epoch_per_rank"] == 926
    assert example["train_steps_total_per_rank"] == 138900
    assert example["target_samples_per_batch_per_gpu"] == 30
    assert abs(example["training_hours"]-19.291666666666668) < 1e-9
    assert abs(example["validation_hours"]-1.6666666666666667) < 1e-9
    two = estimate(train_pairs=13900, val_pairs=3000, mixtures_per_batch=15, world_size=2, epochs=150,
                   train_seconds=1, val_seconds=0.1)
    assert two["train_steps_per_epoch_per_rank"] == 463
    report["time_estimate_arithmetic"] = "passed"
    report["illustrative_seconds_per_step_not_measurements"] = {
        str(seconds): estimate(train_pairs=13900, val_pairs=3000, mixtures_per_batch=15, world_size=1,
                               epochs=150, train_seconds=seconds, val_seconds=0.1)["training_hours"]
        for seconds in (0.1, 0.25, 0.5, 1, 2)}

    # Exercise generator finalization with fake shapes/synchronization, NOT tensors or an optimizer.
    class Shape:
        shape = (30, 1, 48000)
    class FakeCUDA:
        @staticmethod
        def synchronize():
            pass
    class FakeTorch:
        cuda = FakeCUDA()
    timer = TimedLoader([{"wav_mix": Shape()} for _ in range(5)], limit=5, warmup=1, torch_module=FakeTorch())
    for index, batch in enumerate(timer):
        if index == 4:
            break
    timed = timer.finish()
    assert timed["completed_steps"] == 5 and timed["timing"]["count"] == 4
    assert timed["expanded_target_batch_per_gpu"] == 30
    report["timer_counts_last_step_when_executor_breaks"] = True
    assert describe([1, 2, 3])["mean_seconds"] == 2
    report["not_verified"] = ["cloud dependency installation", "CUDA model forward/backward",
                              "150-epoch training", "official evaluation"]
    report["source_sha256"] = {p.name: sha256(p) for p in [PACKAGE/"cloud_entry.py", PACKAGE/"pipeline_worker.py",
                                  PACKAGE/"data_prepare.py", PACKAGE/"evaluation.py", PACKAGE/"EMB_AutoDL_Full_Pipeline.ipynb"]}
    report["status"] = "static_and_data_checks_passed"
    save_json(PACKAGE/"validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
