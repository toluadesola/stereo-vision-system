import threading
import numpy as np
from config import DISTANCE_EMA_ALPHA

# EMA state keyed by (label, object_id) to isolate each tracked instance.
_ema_state: dict[tuple[str, int], float] = {}
_ema_lock = threading.Lock()

_LABEL_STRATEGY: dict[str, tuple[int, float]] = {
    # thin / irregular / low-texture ? use low percentile, tighter ROI
    "person":       (20, 0.4),
    "bicycle":      (20, 0.5),
    "motorbike":    (20, 0.5),
    "cat":          (20, 0.4),
    "dog":          (20, 0.4),
    "chair":        (25, 0.4),
    "bottle":       (20, 0.3),
    "cup":          (20, 0.3),
    "stairs":       (25, 0.5),
    "refrigerator": (50, 0.6),
    "tv":           (50, 0.6),
    "monitor":      (50, 0.6),
    "laptop":       (50, 0.5),
    "microwave":    (50, 0.5),
    "oven":         (50, 0.5),
    "car":          (30, 0.5),
    "bus":          (30, 0.6),
    "truck":        (30, 0.6),
}
_DEFAULT_STRATEGY: tuple[int, float] = (30, 0.5)


def get_object_distance(
    depth_map: np.ndarray,
    box: tuple[int, int, int, int],
    label: str = "",
    object_id: int = 0,
) -> float | None:
    
    key = (label, object_id)
    x, y, w, h = box
    percentile, roi_frac = _LABEL_STRATEGY.get(label, _DEFAULT_STRATEGY)

    # Central ROI shrunk by roi_frac to avoid noisy edges
    margin_x = int(w * (1 - roi_frac) / 2)
    margin_y = int(h * (1 - roi_frac) / 2)
    x1, x2 = x + margin_x, x + w - margin_x
    y1, y2 = y + margin_y, y + h - margin_y

    if x2 <= x1 or y2 <= y1:
        # ROI collapsed (tiny box) ? return existing smoothed value if any
        with _ema_lock:
            return _ema_state.get(key)

    # Clamp to image bounds
    h_map, w_map = depth_map.shape
    x1, x2 = max(0, x1), min(w_map, x2)
    y1, y2 = max(0, y1), min(h_map, y2)

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(~np.isnan(roi)) & (roi > 0.1)]

    if len(valid) == 0:
        # No usable pixels ? return last known smoothed value for this instance
        with _ema_lock:
            return _ema_state.get(key)

    raw = float(np.percentile(valid, percentile))

    with _ema_lock:
        if key not in _ema_state:
            _ema_state[key] = raw
        else:
            _ema_state[key] = (
                DISTANCE_EMA_ALPHA * raw
                + (1.0 - DISTANCE_EMA_ALPHA) * _ema_state[key]
            )
        smoothed = _ema_state[key]

    return smoothed


def reset_ema(label: str | None = None, object_id: int | None = None) -> None:
    
    with _ema_lock:
        if label is None:
            _ema_state.clear()
        elif object_id is None:
            keys_to_delete = [k for k in _ema_state if k[0] == label]
            for k in keys_to_delete:
                del _ema_state[k]
        else:
            _ema_state.pop((label, object_id), None)