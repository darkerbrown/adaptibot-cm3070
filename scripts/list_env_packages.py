"""Simple environment scanner.

Usage:
    python scripts/list_env_packages.py

What it does:
 - Lists installed distributions (pip) and their versions
 - Scans the repository for top-level imports in .py files
 - Reports which imported packages are installed, with versions, and which are missing

This script is safe to run locally and is intended to help reproduce your dev env.
"""
from __future__ import annotations

import sys
import os
import re
from pathlib import Path
from typing import Set, Dict

try:
    # Python 3.8+
    from importlib.metadata import distributions, distribution, PackageNotFoundError
except Exception:
    # Older fallback
    from pkg_resources import working_set as distributions  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]


def list_installed() -> Dict[str, str]:
    """Return mapping of installed package name (lower) -> version."""
    versions: Dict[str, str] = {}
    try:
        for dist in distributions():
            name = getattr(dist, 'metadata', None) and dist.metadata.get('Name') or getattr(dist, 'project_name', None) or getattr(dist, 'key', None) or str(dist)
            ver = getattr(dist, 'version', None) or getattr(dist, 'version', '')
            if name:
                versions[name.lower()] = ver
    except Exception:
        # pkg_resources style
        try:
            import pkg_resources
            for d in pkg_resources.working_set:
                versions[d.project_name.lower()] = d.version
        except Exception:
            pass
    return versions


IMPORT_RE = re.compile(r'^(?:from\s+([\w\.\-]+)\s+import|import\s+([\w\.\-]+))')


def scan_imports(root: Path) -> Set[str]:
    """Scan .py files for top-level import names (first module segment)."""
    names: Set[str] = set()
    for p in root.rglob('*.py'):
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        for line in text.splitlines():
            m = IMPORT_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1) or m.group(2)
            if name:
                top = name.split('.')[0].lower()
                # ignore standard library shortcuts (a conservative list)
                if top in ('os','sys','re','math','time','json','argparse','dataclasses','typing','pathlib','itertools','collections','heapq'):
                    continue
                names.add(top)
    return names


def main():
    print(f"Scanning repository at: {REPO_ROOT}")
    installed = list_installed()
    imports = scan_imports(REPO_ROOT)

    print('\nInstalled packages (sample):')
    sample = list(installed.items())[:30]
    for k, v in sample:
        print(f"  {k}=={v}")

    print('\nImports found in repo:')
    for name in sorted(imports):
        ver = installed.get(name)
        status = f"installed=={ver}" if ver else "MISSING"
        print(f"  {name}: {status}")

    missing = [n for n in sorted(imports) if n not in installed]
    print('\nSummary:')
    print(f"  total imports found: {len(imports)}")
    print(f"  missing packages: {len(missing)}")
    if missing:
        print('  Missing package names:')
        for m in missing:
            print(f"    - {m}")


if __name__ == '__main__':
    main()
