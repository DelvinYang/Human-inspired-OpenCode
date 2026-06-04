from __future__ import annotations

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_building import select_individual_loro_localized_only as localized


localized.LABELS = {
    "INTERACTION_to_US": "INTERACTION -> US",
    "INTERACTION_to_CN": "INTERACTION -> CN",
    "INTERACTION_to_DE": "INTERACTION -> DE",
}


if __name__ == "__main__":
    localized.main()
