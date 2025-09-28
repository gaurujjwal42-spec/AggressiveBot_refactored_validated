import hashlib, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / 'aggbot' / 'legacy'
OUT = Path(__file__).resolve().parents[1] / 'aggbot' / 'legacy_dedup'
REPORT = Path(__file__).resolve().parents[1] / 'LEGACY_DEDUP_REPORT.txt'
def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    entries = []
    for p in ROOT.rglob('*'):
        if p.is_file():
            hx = file_hash(p)
            if hx not in hashes:
                hashes[hx] = p
                rel = p.relative_to(ROOT)
                tgt = OUT / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
            entries.append((hx, str(p)))
    with open(REPORT, 'w') as w:
        w.write('Legacy de-duplication report\n')
        w.write(f'Unique files: {len(hashes)}\n')
        w.write(f'Total files:  {len(entries)}\n\n')
        groups = {}
        for hx, path in entries:
            groups.setdefault(hx, []).append(path)
        for hx, paths in groups.items():
            if len(paths) > 1:
                w.write(f'DUP {hx[:12]}\n')
                for p in paths:
                    w.write(f'  - {p}\n')
    print('Done. Report at', REPORT)
if __name__ == '__main__':
    main()
