#!/bin/sh
# Build all Mira native components (crate-less Rust + C/GTK3) into ./build/bin
# without touching the live ~/.config session.
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
OUT_DIR="$ROOT/build"
BIN_DIR="$OUT_DIR/bin"
CARGO_TARGET="${XDG_CACHE_HOME:-$HOME/.cache}/mira-dev/cargo-target"
UI_OUT="$OUT_DIR/misa-ui"

mkdir -p "$BIN_DIR" "$CARGO_TARGET" "$UI_OUT"

for crate in misa-clip misa-backdrop misa-wall misa-scroll; do
    manifest="$ROOT/src/$crate/Cargo.toml"
    [ -f "$manifest" ] || continue
    echo ">>> Building $crate (Rust, offline)..."
    cargo build --quiet --release --offline \
        --manifest-path "$manifest" \
        --target-dir "$CARGO_TARGET"
done

for bin in misa-clip misa-backdrop misa-wall Scroll; do
    if [ -f "$CARGO_TARGET/release/$bin" ]; then
        cp -f "$CARGO_TARGET/release/$bin" "$BIN_DIR/.$bin.new"
        mv -f "$BIN_DIR/.$bin.new" "$BIN_DIR/$bin"
        chmod 755 "$BIN_DIR/$bin"
    fi
done

if [ -f "$ROOT/src/misa-ui/Makefile" ]; then
    echo ">>> Building misa-ui (C + GTK3)..."
    make --quiet -C "$ROOT/src/misa-ui" OUT="$UI_OUT"
    for tool in "$UI_OUT"/*; do
        [ -f "$tool" ] && [ -x "$tool" ] || continue
        name=${tool##*/}
        cp -f "$tool" "$BIN_DIR/.$name.new"
        mv -f "$BIN_DIR/.$name.new" "$BIN_DIR/$name"
        chmod 755 "$BIN_DIR/$name"
    done
fi

# Optional sibling x-shot screenshot overlay
SHOT_SRC="${X_SHOT_SRC:-$ROOT/../x-shot}"
if [ -f "$SHOT_SRC/Cargo.toml" ] && command -v cargo >/dev/null 2>&1; then
    echo ">>> Building x-shot (Rust, offline)..."
    if cargo build --quiet --release --offline \
        --manifest-path "$SHOT_SRC/Cargo.toml" \
        --target-dir "$CARGO_TARGET"; then
        cp -f "$CARGO_TARGET/release/x-shot" "$BIN_DIR/.x-shot.new"
        mv -f "$BIN_DIR/.x-shot.new" "$BIN_DIR/x-shot"
        chmod 755 "$BIN_DIR/x-shot"
    fi
fi

chmod 755 "$ROOT/session/bspwmrc" "$ROOT/session/session.sh" \
    "$ROOT"/session/bin/* "$ROOT"/session/bars/*/scripts/* \
    "$ROOT"/tools/*.sh 2>/dev/null || true

echo ">>> Build complete: $(ls "$BIN_DIR" | tr '\n' ' ')"
