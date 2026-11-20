#!/usr/bin/env python3
"""Replay an archived analysis script in an isolated, repository-local workspace.

Reconstructs the original virtual directory tree under `.runtime/<target>/project`
by linking shared primary data (`data/`) and archived intermediate tables via
`manifests/file_manifest.csv`, and adapts the original machine-specific project root
in a temporary script copy without modifying archived files.
Inspect the command first with `--dry-run`.
"""
from pathlib import Path
import argparse
import csv
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
OLD_ROOT = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('module', help='Analysis module number (1-15) or "manuscript"')
    ap.add_argument('script', help='Script filename or unique path suffix')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    args, extra = ap.parse_known_args()

    if args.module.lower() == 'manuscript':
        mod_key = 'Manuscript'
        runtime_name = 'manuscript'
    else:
        try:
            mnum = int(args.module)
            assert 1 <= mnum <= 15
        except (ValueError, AssertionError):
            sys.exit('module must be an integer 1..15 or "manuscript"')
        mod_key = f'Analysis {mnum:02d}'
        runtime_name = f'analysis_{mnum:02d}'

    manifest_path = REPO / 'manifests/file_manifest.csv'
    with manifest_path.open(encoding='utf-8') as fh:
        manifest = list(csv.DictReader(fh))

    mod_rows = [r for r in manifest if r['module'] == mod_key and r['source_path']]
    candidates = [
        r for r in mod_rows
        if ('/code/' in r['repo_path'] or r['repo_path'].startswith('scripts/'))
        and r['source_path'].endswith(args.script)
    ]
    if len(candidates) != 1:
        sys.exit('Expected one matching script; found %d.\n%s' %
                 (len(candidates), '\n'.join(r['source_path'] for r in candidates)))
    script = candidates[0]

    # Map every archived source_path into the isolated virtual project root
    by_source = {}
    for r in manifest:
        if r['source_path'] and r['module'] != mod_key:
            by_source.setdefault(r['source_path'], r)
    for r in mod_rows:
        by_source[r['source_path']] = r

    work = REPO / '.runtime' / runtime_name
    project = work / 'project'
    if work.exists():
        shutil.rmtree(work)
    for r in by_source.values():
        target = project / r['source_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPO / r['repo_path']
        if target.exists() or target.is_symlink():
            target.unlink()
        if r['module'] == mod_key and ('/code/' in r['repo_path'] or '/results/' in r['repo_path'] or '/notes/' in r['repo_path'] or r['repo_path'].startswith('scripts/')):
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)

    original = REPO / script['repo_path']
    configured = project / script['source_path']
    text = original.read_text(encoding='utf-8')
    if OLD_ROOT not in text:
        print('Note: script has no original absolute root; inspect relative paths before running.')
    configured.write_text(text.replace(OLD_ROOT, str(project)), encoding='utf-8')

    suffix = configured.suffix
    if suffix == '.R':
        cmd = ['Rscript', str(configured), *extra]
    elif suffix in {'.sh', '.bash'}:
        cmd = ['bash', str(configured), *extra]
    else:
        cmd = [args.python, str(configured), *extra]
    print('Configured script:', configured)
    print('Isolated project root:', project)
    print('Command:', ' '.join(cmd))
    if not args.dry_run:
        return subprocess.run(cmd, cwd=configured.parent).returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
