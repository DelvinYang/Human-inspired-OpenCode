from cultural_align.training.data import NormStats, compute_stats, load_dataset_split, load_multi_split

__all__ = [
    "NormStats",
    "compute_stats",
    "load_dataset_split",
    "load_multi_split",
    "train_supervised",
    "train_transfer",
    "evaluate_checkpoint",
]


def __getattr__(name: str):
    if name in {"evaluate_checkpoint", "train_supervised", "train_transfer"}:
        from cultural_align.training import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
