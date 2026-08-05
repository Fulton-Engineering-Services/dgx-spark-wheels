#!/usr/bin/env python3
"""
Generate a PEP 503 (simple repository API) package index from packages.json,
pointing each entry at its corresponding GitHub Release asset.

Usage:
    scripts/generate-index.py --repo <owner>/<repo> --out-dir public/simple

Produces:
    public/simple/index.html                  (root index, links to each package)
    public/simple/<package>/index.html         (per-package index, links to each wheel)

Designed to be run in CI (see .github/workflows/publish-index.yml) and the
output published via GitHub Pages, so:

    pip install <package> --extra-index-url https://<owner>.github.io/<repo>/simple/

resolves correctly. See https://peps.python.org/pep-0503/.
"""

import argparse
import html
import json
import pathlib
import sys

INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
  <head><meta charset="utf-8"><title>{title}</title></head>
  <body>
{body}
  </body>
</html>
"""


def normalize(name: str) -> str:
    """PEP 503 name normalization: lowercase, runs of -_. collapsed to a single -."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def build_index(packages: list[dict], repo: str, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    root_links = []
    by_name: dict[str, list[dict]] = {}
    for pkg in packages:
        by_name.setdefault(pkg["name"], []).append(pkg)

    for name, versions in sorted(by_name.items()):
        norm = normalize(name)
        pkg_dir = out_dir / norm
        pkg_dir.mkdir(parents=True, exist_ok=True)

        links = []
        for pkg in versions:
            wheel = pkg["wheel"]
            version = pkg["version"]
            # Release tags use the PUBLIC version segment only (no "+local"),
            # so "+" never lands in a git tag or download URL. The local-version
            # segment (e.g. +cu13.0torch2.13.glibc235) lives in the wheel
            # filename/metadata, not the tag. Prefer an explicit
            # "public_version" field; fall back to stripping "+...".
            public_version = pkg.get("public_version") or version.split("+", 1)[0]
            url = f"https://github.com/{repo}/releases/download/{name}-v{public_version}/{wheel}"
            links.append(f'    <a href="{html.escape(url)}">{html.escape(wheel)}</a><br/>')

        (pkg_dir / "index.html").write_text(
            INDEX_TEMPLATE.format(
                title=f"Links for {name}",
                body="\n".join(links),
            )
        )
        root_links.append(f'    <a href="{norm}/">{html.escape(name)}</a><br/>')

    (out_dir / "index.html").write_text(
        INDEX_TEMPLATE.format(title="dgx-spark-wheels", body="\n".join(root_links))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="owner/repo, e.g. Fulton-Engineering-Services/dgx-spark-wheels",
    )
    parser.add_argument("--manifest", default="packages.json", type=pathlib.Path)
    parser.add_argument("--out-dir", default="public/simple", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    packages = manifest["packages"]
    build_index(packages, args.repo, args.out_dir)
    print(f"Wrote index for {len(packages)} package(s) to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
