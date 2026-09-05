#!/usr/bin/env python3
"""Discover all pip packages needed to import a target module, in one shot.

Usage: python3 discover_import_deps.py "lmcache.v1.multiprocess.server.MPCacheServer"

Runs pip install with --dry-run --report to find the closure of all needed
packages at once (no 20-miniute CI round-trips).
"""

import json
import subprocess
import sys


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "lmcache"
    pkg_name = target.split(".")[0]

    # 1. Try importing to collect missing modules
    missing: list[str] = []
    while True:
        code = f"import {target}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break

        stderr = result.stderr
        if "ModuleNotFoundError" not in stderr:
            print(f"Unexpected error:\n{stderr}", file=sys.stderr)
            sys.exit(1)

        mod = None
        for line in stderr.split("\n"):
            if "No module named" in line:
                mod = line.split("'")[1]
                break

        if mod is None or mod in missing:
            print(f"Cannot ressolve:\n{stderr}", file=sys.stderr)
            sys.exit(1)

        missing.append(mod)
        print(f"Missing: {mod}")

    if not missing:
        print(f"All deps already satisfieed for `import {target}`")
        return

    # 2. Use pip's dry-run to find the full closure (handles pip-name != import-name)
    existing = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    ).stdout
    have = set(line.split("==")[0].lower() for line in existing.strip().split("\n") if "==" in line)

    all_deps = list(missing)
    idx = 0
    while idx < len(all_deps):
        mod = all_deps[idx]
        idx += 1
        if mod in have:
            continue
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--report", "-", mod],
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout) if result.returncode == 0 else {}
            for entry in report.get("install", []):
                name = entry["metadata"]["name"].lower()
                if name not in have and name != pkg_name:
                    all_deps.append(name)
                have.add(name)
        except Exception as e:
            print(f"Warning: failed to resolve {mod}: {e}", file=sys.stderr)

    # Deduplicate keeping order
    seen: set[str] = set()
    unique: list[str] = []
    for d in all_deps:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    print(f"\nInstall with: pip install {' '.join(unique)}")
    print(f"\nOr paste into your script:\npip install {' '.join(unique)}")


if __name__ == "__main__":
    main()
