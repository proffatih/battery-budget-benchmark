#!/usr/bin/env bash
# Download and unpack the CALCE cells used in the study.
# Source: https://web.calce.umd.edu/batteries/data.htm  (CALCE terms of use apply)
set -eu
DEST="${1:-data/calce}"
mkdir -p "$DEST/ex"
for c in CS2_35 CS2_36 CS2_37 CS2_38 CX2_16 CX2_34 CX2_36 CX2_37; do
  if [ ! -s "$DEST/$c.zip" ]; then
    echo "downloading $c"
    curl -sSL "https://web.calce.umd.edu/batteries/data/$c.zip" -o "$DEST/$c.zip"
  fi
  unzip -o -q "$DEST/$c.zip" -d "$DEST/ex"
done
echo "done -> $DEST/ex"
