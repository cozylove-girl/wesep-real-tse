"""Pure arithmetic; importing this module never loads a model or trains."""
import math
import statistics


def describe(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be finite and positive")
    ordered = sorted(samples)
    return {"count": len(samples), "mean_seconds": statistics.mean(samples),
            "median_seconds": statistics.median(samples),
            "p90_seconds": ordered[min(len(ordered)-1, math.ceil(len(ordered)*0.9)-1)]}


def estimate(*, train_pairs, val_pairs, mixtures_per_batch, world_size, epochs,
             train_seconds, val_seconds, completed_epochs=0):
    if min(train_pairs, val_pairs, mixtures_per_batch, world_size, epochs) <= 0:
        raise ValueError("Counts must be positive")
    train_steps = train_pairs//world_size//mixtures_per_batch
    val_steps = val_pairs//world_size//mixtures_per_batch
    if min(train_steps, val_steps) == 0:
        raise ValueError("Batch size/world size leaves zero steps")
    # Match upstream trainer's floor divisions. Collate doubles target batch,
    # not step count, because each mixture provides two target speakers.
    train_hours = train_steps*epochs*train_seconds/3600
    val_hours = val_steps*epochs*val_seconds/3600
    remaining = max(0, epochs-completed_epochs)
    return {"train_pairs": train_pairs, "val_pairs": val_pairs,
            "mixtures_per_batch_per_gpu": mixtures_per_batch,
            "target_samples_per_batch_per_gpu": mixtures_per_batch*2,
            "world_size": world_size, "epochs": epochs,
            "train_steps_per_epoch_per_rank": train_steps, "val_steps_per_epoch_per_rank": val_steps,
            "train_steps_total_per_rank": train_steps*epochs,
            "train_seconds_per_step": train_seconds, "val_seconds_per_step": val_seconds,
            "training_hours": train_hours, "validation_hours": val_hours,
            "train_and_validation_hours": train_hours+val_hours,
            "remaining_train_and_validation_hours": remaining*(train_steps*train_seconds+val_steps*val_seconds)/3600,
            "excluded": ["environment installation", "downloads/data generation", "checkpoint writes", "model averaging", "final REAL-T evaluation"],
            "notice": "A measured extrapolation, not a guarantee. DDP uses slowest-rank mean; validation and train costs differ."}
