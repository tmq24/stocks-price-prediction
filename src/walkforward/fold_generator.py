from dataclasses import dataclass


@dataclass
class Fold:
    """
    Row-index boundaries for one chronological (70/30) split fold.
    All indices are relative to a sorted per-ticker DataFrame.
    start: inclusive, end: exclusive (Python slice convention).
    """
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
