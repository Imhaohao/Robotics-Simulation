#!/usr/bin/env bash
# Pin the MuJoCo Menagerie commit that contains Franka FR3 and Robotiq 2F-85.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/mujoco_menagerie"
PIN="370c43965323c0944b433b202aac5058b5f907a9"
if [[ -f "$DEST/franka_fr3/fr3.xml" && -f "$DEST/robotiq_2f85/2f85.xml" ]]; then
  echo "assets already present at $DEST"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$ROOT/third_party"
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set franka_fr3 robotiq_2f85
git -C "$DEST" fetch --depth 1 origin "$PIN"
git -C "$DEST" checkout "$PIN"
echo "fetched $PIN"
