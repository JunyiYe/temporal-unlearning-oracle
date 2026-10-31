"""Delete every artifact of an architecture whose hyperparameters changed.

Two defaults in era_pilot make a config change dangerous: --skip-existing
leaves stale cells in place, and an existing oracle checkpoint is LOADED
rather than retrained. Both would silently mix hyperparameter generations.
Run this before re-running any architecture whose config was edited.

    python scripts/purge_stale_cells.py --arch patchtst            # dry run
    python scripts/purge_stale_cells.py --arch patchtst --apply
    python scripts/purge_stale_cells.py --auto --apply             # all drifted
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

ROOT = Path('/nfs/storage1/home/ye/fin-unlearning')
OUT = ROOT / 'v2/outputs'
MANIFEST = OUT / 'config_manifest.json'


def targets(arch: str) -> list[Path]:
    stem = f'{arch}_sp500_vol_v2'
    hits = list((OUT / 'pilot').glob(f'{stem}_*_unlearn.json'))
    hits += list((OUT / 'pilot' / 'ckpt').glob(f'{stem}_*.pt'))
    hits += list((OUT / 'sweep').glob(f'{stem}_fold*_model.pt'))
    hits += list((OUT / 'detector').glob(f'{stem}_*'))
    return sorted(p for p in hits if p.exists())


def drifted() -> list[str]:
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from config_fingerprint import current
    old = json.loads(MANIFEST.read_text())['fingerprints'] if MANIFEST.exists() else {}
    return [a for a, f in current().items() if old.get(a) not in (None, f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch')
    ap.add_argument('--auto', action='store_true', help='purge every drifted arch')
    ap.add_argument('--apply', action='store_true', help='actually delete')
    args = ap.parse_args()

    archs = drifted() if args.auto else ([args.arch] if args.arch else [])
    if not archs:
        print('nothing to purge'); return
    for arch in archs:
        files = targets(arch)
        print(f'\n{arch}: {len(files)} artifacts')
        for p in files[:5]:
            print(f'    {p.relative_to(OUT)}')
        if len(files) > 5:
            print(f'    ... and {len(files)-5} more')
        if args.apply:
            trash = OUT / 'purged' / arch
            trash.mkdir(parents=True, exist_ok=True)
            for p in files:
                dest = trash / p.name
                shutil.move(str(p), str(dest)) if p.is_file() else shutil.rmtree(p)
            print(f'  -> moved to {trash} (not deleted; remove by hand when sure)')
        else:
            print('  (dry run; pass --apply)')


if __name__ == '__main__':
    main()
