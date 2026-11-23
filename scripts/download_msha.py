from __future__ import annotations
import argparse, hashlib, json, os, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'external' / 'msha'
URL = 'https://arlweb.msha.gov/opengovernmentdata/DataSets/Accidents.zip'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def extract_safely(archive: Path, target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            rel = PurePosixPath(entry.filename)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            out = (target / Path(*rel.parts)).resolve()
            if root not in out.parents and out != root:
                raise ValueError(f'Archive member escapes target: {entry.filename}')
            if entry.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src, out.open('wb') as dst:
                for chunk in iter(lambda: src.read(1 << 20), b''):
                    dst.write(chunk)
            extracted.append(out)
    return extracted

def main() -> None:
    parser = argparse.ArgumentParser(description='Download official public MSHA accident data.')
    parser.add_argument('--refresh', action='store_true', help='download a fresh snapshot')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / 'Accidents.zip'
    if args.refresh or not archive.exists():
        request = urllib.request.Request(URL, headers={
            'User-Agent': 'SIF-Precursor-Research/1.0',
            'Referer': 'https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp',
        })
        temp = archive.with_suffix('.zip.part')
        try:
            with urllib.request.urlopen(request, timeout=180) as response, temp.open('wb') as sink:
                while chunk := response.read(1 << 20):
                    sink.write(chunk)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        os.replace(temp, archive)
    files = extract_safely(archive, DEST / 'extracted')
    accidents = next((p for p in files if p.name.casefold() == 'accidents.txt'), DEST / 'extracted' / 'Accidents.txt')
    if not accidents.exists():
        raise FileNotFoundError('Accidents.txt is absent from the official archive.')
    manifest = {
        'source_url': URL,
        'source_page': 'https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp',
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive': archive.name,
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': sha256(archive),
        'extracted_file': accidents.relative_to(ROOT).as_posix(),
        'note': 'The agency updates the source periodically; retain this manifest with reported metrics.',
    }
    (DEST / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
