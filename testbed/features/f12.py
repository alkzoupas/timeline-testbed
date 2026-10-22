"""F12: the 12 frozen F-* detector features, as their raw values (SPEC §6.2).

Read from the latest payload of the unit (the one with the latest `window_start`, SPEC §6.1).
The set of ids is checked on every unit, so a detector change shows up as an error, not a
silently shifted column.
"""

from __future__ import annotations

import math

from testbed.pipeline.units import Unit

F_IDS = (
    "F-AGE-1",
    "F-CON-1",
    "F-CYC-1",
    "F-CYC-2",
    "F-CYC-3",
    "F-CYC-4",
    "F-DEV-1",
    "F-DEV-2",
    "F-DOM-1",
    "F-DOM-2",
    "F-PAY-1",
    "F-VER-1",
)


def f12(unit: Unit) -> dict[str, float]:
    """`raw_value` of each F-* contribution in the unit's latest payload, keyed by feature id."""
    contributions = unit.latest["contributions"]
    values = {c["feature_id"]: float(c["raw_value"]) for c in contributions}
    if set(values) != set(F_IDS) or len(contributions) != len(F_IDS):
        raise ValueError(f"unit {unit.snapshot_id}: F-* set changed: {sorted(values)}")
    for feature_id, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"unit {unit.snapshot_id}: {feature_id} is not finite")
    return {feature_id: values[feature_id] for feature_id in F_IDS}
