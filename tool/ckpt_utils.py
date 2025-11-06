from typing import Iterable, Mapping, MutableMapping, Optional, Set
import torch

def _unwrap_state_dict(obj):
    """
    Accept checkpoints saved as either a raw state_dict or a dict with
    'model' / 'state_dict' keys.
    """
    if isinstance(obj, Mapping):
        for k in ("model", "state_dict", "module"):
            if k in obj and isinstance(obj[k], Mapping):
                return obj[k]
    return obj

def load_state_dict_safely(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    map_location: str | torch.device = "cpu",
    drop_exact_keys: Optional[Iterable[str]] = None,
    drop_prefixes: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Load a checkpoint while *silently* dropping keys that do not exist in the model
    or are explicitly marked for removal. This avoids printing `_IncompatibleKeys`.

    Typical use for GroundingDINO:
        drop_exact_keys = {"label_enc.weight", "bert.embeddings.position_ids"}
    """
    drop_exact: Set[str] = set(drop_exact_keys or [])
    drop_pref: Set[str] = set(drop_prefixes or [])

    raw = torch.load(ckpt_path, map_location=map_location)
    state: MutableMapping[str, torch.Tensor] = dict(_unwrap_state_dict(raw))
    model_state = model.state_dict()

    # Filter out keys we know we don't need or don't have in the model
    filtered = {}
    filtered_out = []
    for k, v in state.items():
        if k in drop_exact:
            filtered_out.append(k)
            continue
        if any(k.startswith(p) for p in drop_pref):
            filtered_out.append(k)
            continue
        if k not in model_state:
            filtered_out.append(k)
            continue
        filtered[k] = v

    # Load filtered state dict
    msg = model.load_state_dict(filtered, strict=False)

    if verbose:
        missing = list(msg.missing_keys)
        unexpected = list(msg.unexpected_keys)
        # unexpected should normally be empty because we filtered, but keep for completeness
        summary = (
            f"Checkpoint loaded: {ckpt_path}\n"
            f" - applied params: {len(filtered)}\n"
            f" - dropped (not in model or by rule): {len(filtered_out)}\n"
            f" - still missing in model: {len(missing)}\n"
            f" - still unexpected: {len(unexpected)}"
        )
        print(summary)