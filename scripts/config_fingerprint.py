"""Fingerprint the model/train sections of every v2 config.

Cells are only comparable if they were produced by the same hyperparameters.
The cell JSONs record the config *path*, not its contents, so a config edit
would silently mix generations. This writes a manifest of the current
fingerprints; purge_stale_cells.py diffs against it.

    python scripts/config_fingerprint.py            # write/refresh manifest
    python scripts/config_fingerprint.py --check    # exit 1 if any drifted
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import yaml

ROOT = Path('/nfs/storage1/home/ye/fin-unlearning')
CONFIGS = ROOT / 'v2/configs'
MANIFEST = ROOT / 'v2/outputs/config_manifest.json'


def fingerprint(path: Path) -> str:
    cfg = yaml.safe_load(path.read_text())
    payload = json.dumps({k: cfg.get(k) for k in ('model', 'train', 'task')},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def current() -> dict[str, str]:
    return {p.stem.replace('_sp500_vol_v2', ''): fingerprint(p)
            for p in sorted(CONFIGS.glob('*_sp500_vol_v2.yaml'))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    now = current()
    if args.check:
        if not MANIFEST.exists():
            print('no manifest; nothing to compare'); return
        old = json.loads(MANIFEST.read_text())['fingerprints']
        drift = {a: (old.get(a), f) for a, f in now.items() if old.get(a) not in (None, f)}
        for a, (o, f) in drift.items():
            print(f'DRIFT {a}: {o} -> {f}  (purge its cells before re-running)')
        for a in now:
            if a not in old:
                print(f'NEW   {a}: {now[a]}')
        sys.exit(1 if drift else 0)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({'fingerprints': now}, indent=2, sort_keys=True))
    for a, f in now.items():
        print(f'{a:14s} {f}')
    print(f'\nwrote {MANIFEST}')


if __name__ == '__main__':
    main()
