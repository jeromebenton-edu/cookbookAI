"""Convert labeled recipes into LayoutLMv3 training format."""

from pathlib import Path
from typing import List


def build_dataset(label_dir: Path) -> List[dict]:
    """
    TODO: Convert labeled line data into token-level annotations.
    """
    _ = label_dir
    return []


if __name__ == "__main__":
    print("TODO: implement dataset builder")
