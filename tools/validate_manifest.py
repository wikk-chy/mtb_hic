#!/usr/bin/env python3
"""Validate repository integrity, SHA-256 checksums, evidence exclusions, and file-size bounds."""
from pathlib import Path
import csv
import hashlib
import json
import sys

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / 'manifests/file_manifest.csv'
REPORT = REPO / 'manifests/validation_report.json'
DISALLOWED_EXT = {'.pdf', '.png', '.jpg', '.jpeg', '.svg', '.ai', '.eps', '.tif', '.tiff', '.tex', '.cls', '.bst'}
ALLOWED_README_ASSETS = {
    'assets/readme_banner.png',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    errors = []
    with MANIFEST.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    paths = [row['repo_path'] for row in rows]
    if len(paths) != len(set(paths)):
        errors.append('Duplicate repo_path entries in manifest')
    for row in rows:
        p = REPO / row['repo_path']
        if not p.is_file():
            errors.append(f"Missing: {row['repo_path']}")
            continue
        if p.stat().st_size != int(row['bytes']):
            errors.append(f"Size mismatch: {row['repo_path']}")
        if sha256(p) != row['sha256']:
            errors.append(f"SHA-256 mismatch: {row['repo_path']}")
        if p.suffix.lower() in DISALLOWED_EXT and row['repo_path'] not in ALLOWED_README_ASSETS:
            errors.append(f"Disallowed binary/manuscript extension tracked: {row['repo_path']}")
    # Check retired CID exclusions in analysis_11
    a11 = [p for p in paths if p.startswith('response_analyses/analysis_11_')]
    forbidden = ('whole_cid', 'fig5_whole', 'rescaled_pileup', 'fig_CID_validation_merged',
                 'fig1e_cid_coexpression', 'fig_combined_CID_validation')
    for p in a11:
        if any(tok in p for tok in forbidden):
            errors.append(f'Superseded CID evidence retained in analysis_11: {p}')
    # Check file size limit (< 50 MiB warning / < 100 MiB GitHub hard limit)
    oversized = [p for p in paths if (REPO / p).is_file() and (REPO / p).stat().st_size > 100 * 1024**2]
    if oversized:
        errors.append('Files exceeding 100 MiB: ' + ', '.join(oversized))
    # Verify complete manifest coverage
    manifest_set = set(paths)
    skip = {'manifests/file_manifest.csv', 'manifests/validation_report.json'}
    untracked = []
    for p in REPO.rglob('*'):
        if not p.is_file() or '.git' in p.parts or '.runtime' in p.parts or p.name == '.DS_Store':
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel not in manifest_set and rel not in skip:
            untracked.append(rel)
    if untracked:
        errors.append(f'Untracked files not in manifest ({len(untracked)}): ' + ', '.join(untracked[:10]))

    result = {
        'status': 'PASS' if not errors else 'FAIL',
        'manifest_files': len(rows),
        'manifest_bytes': sum(int(r['bytes']) for r in rows),
        'analysis_modules': 15,
        'errors': errors,
        'checks': [
            'file existence', 'file size', 'SHA-256', 'unique paths',
            'no figure/manuscript binaries', 'analysis_11 CID exclusions',
            '100 MiB per-file limit', 'complete manifest coverage',
        ],
    }
    REPORT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
