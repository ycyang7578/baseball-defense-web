"""Train the speed+cos+sin+fielder_dist model → models/2025/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training import train_position

YEARS = [2021, 2022, 2023, 2024]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "2025"

if __name__ == "__main__":
    positions = sys.argv[1:] or ["LF", "CF", "RF"]
    for pos in positions:
        train_position(pos, YEARS, OUTPUT_DIR)
