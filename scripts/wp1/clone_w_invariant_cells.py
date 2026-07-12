"""Clone W-invariant injection JSONs (cascade uses W_primary=120 only).

For cells where only W differs, copy results from W=120 sibling and restamp metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.wp1.signal_injector import hash_combine

INJ = _REPO / "data" / "injection_runs"
BOOT_SEED = 43

CLONES = [
    (0.15, 5, 120, 240),
]


def main() -> int:
    for delta, q, w_src, w_dst in CLONES:
        src = INJ / f"inj_d{delta}_q{q}_W{w_src}.json"
        dst = INJ / f"inj_d{delta}_q{q}_W{w_dst}.json"
        if dst.exists():
            print(f"skip existing {dst.name}")
            continue
        if not src.exists():
            print(f"missing source {src.name}", file=sys.stderr)
            return 1
        data = json.loads(src.read_text(encoding="utf-8"))
        grid_hash = hash_combine(BOOT_SEED, int(w_dst), int(q) * 1000 + int(delta * 100000))
        data["W"] = w_dst
        data["grid_hash"] = grid_hash
        prov = data.setdefault("provenance", {})
        prov["w_invariant_clone"] = True
        prov["cloned_from"] = src.name
        dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"cloned {src.name} -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
