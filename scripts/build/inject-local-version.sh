#!/usr/bin/env bash
# inject-local-version.sh <wheel> <local_seg>
#
# Rewrites a built wheel's version to <public_version>+<local_seg> (PEP 440
# local version) so the filename + METADATA become self-diagnosing for the
# real ABI (CUDA/torch/glibc) the wheel was built against — since the
# cp312-linux_aarch64 tag encodes none of that.
#
# Approach: unpack with `wheel unpack`, edit .dist-info/METADATA Version:,
# rename the .dist-info dir to the new version, then `wheel pack` (which
# regenerates RECORD with correct hashes and names the file from METADATA).
# Idempotent: if the version already carries the segment, copies as-is.
#
# The repacked wheel lands in ./dist/ (relative to cwd). Prints the path of
# the resulting wheel to stdout.

set -euo pipefail

wheel="$1"
local_seg="$2"

[ -f "$wheel" ] || { echo "inject-local-version: wheel not found: $wheel" >&2; exit 1; }
[ -n "$local_seg" ] || { echo "inject-local-version: empty local segment" >&2; exit 1; }

mkdir -p dist
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

python3 -m wheel unpack "$wheel" -d "$tmpdir/unpack" >&2

unpack_root="$(find "$tmpdir/unpack" -mindepth 1 -maxdepth 1 -type d | head -1)"
[ -n "$unpack_root" ] || { echo "inject-local-version: unpack produced no dir" >&2; exit 1; }

distinfo="$(find "$unpack_root" -maxdepth 1 -name '*.dist-info' | head -1)"
[ -n "$distinfo" ] || { echo "inject-local-version: no .dist-info found" >&2; exit 1; }

metadata="$distinfo/METADATA"
old_version="$(grep -m1 '^Version: ' "$metadata" | sed 's/^Version: //')"
public_version="${old_version%%+*}"
new_version="${public_version}+${local_seg}"

if [ "$old_version" = "$new_version" ]; then
  echo "inject-local-version: $wheel already at $new_version; copying as-is" >&2
  cp "$wheel" "dist/$(basename "$wheel")"
  echo "dist/$(basename "$wheel")"
  exit 0
fi

# Rewrite the Version line in METADATA.
sed -i "s/^Version: ${old_version}\$/Version: ${new_version}/" "$metadata"

# Rename the .dist-info directory to the new version.
old_di="$(basename "$distinfo")"
new_di="${old_di/${old_version}/${new_version}}"
mv "$distinfo" "$unpack_root/$new_di"

# Repack: wheel pack regenerates RECORD and names the file from METADATA.
python3 -m wheel pack "$unpack_root" -d dist/ >&2

out="$(ls -t dist/*.whl | head -1)"
echo "inject-local-version: wrote $out" >&2
echo "$out"
