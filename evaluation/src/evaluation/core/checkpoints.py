from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch


def load_finetuned_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Finetuned checkpoint must be a dictionary-like .pth file.")

    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Unable to find model state_dict in finetuned checkpoint.")

    if any(str(key).startswith("backbone.") for key in state_dict):
        backbone_state = {
            str(key).removeprefix("backbone."): value
            for key, value in state_dict.items()
            if str(key).startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError("Checkpoint has no backbone weights to load.")
        return backbone_state
    return cast(dict[str, Any], state_dict)
