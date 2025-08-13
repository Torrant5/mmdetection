#!/usr/bin/env bash
set -euo pipefail

# 用途: リポジトリ直下から、外部USB上の data/dataset/result へシンボリックリンクを作成
# 既定のマウントポイントは /Volumes/kohssd。環境変数で上書き可。

USB_ROOT=${USB_ROOT:-/Volumes/kohssd}
# 相対指定された場合でも動作するよう補正（例: "Volumes/kohssd" → "/Volumes/kohssd"）
case "$USB_ROOT" in
  /*) ;; # absolute path, ok
  *) USB_ROOT="/${USB_ROOT}" ;;
esac
LINK_OUTPUT=${LINK_OUTPUT:-1}      # output -> ${USB_ROOT}/result
LINK_DATASETS=${LINK_DATASETS:-1}  # data/<name> -> ${USB_ROOT}/dataset/<name>
LINK_RAW=${LINK_RAW:-1}            # data/raw -> ${USB_ROOT}/data
FORCE=${FORCE:-0}                  # 既存の衝突時に .bak 退避して上書き

ts() { date +%Y%m%d_%H%M%S; }

link_path() {
  local target="$1"; shift
  local linkname="$1"; shift

  if [[ -L "$linkname" ]]; then
    local current
    current=$(readlink "$linkname") || current=""
    if [[ "$current" == "$target" ]]; then
      echo "[OK] Symlink exists: $linkname -> $target"
      return 0
    else
      if [[ "$FORCE" == "1" ]]; then
        mv "$linkname" "${linkname}.bak_$(ts)"
        echo "[INFO] Replacing existing symlink: $linkname (backup created)"
      else
        echo "[ERR] $linkname already exists and points to $current. Set FORCE=1 to replace." >&2
        return 1
      fi
    fi
  elif [[ -e "$linkname" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      mv "$linkname" "${linkname}.bak_$(ts)"
      echo "[INFO] Backed up existing path: $linkname"
    else
      echo "[ERR] $linkname already exists. Set FORCE=1 to backup and replace." >&2
      return 1
    fi
  fi

  ln -s "$target" "$linkname"
  echo "[NEW] $linkname -> $target"
}

if [[ ! -d "$USB_ROOT" ]]; then
  echo "[ERR] USB root not found: $USB_ROOT (override with USB_ROOT=/path)" >&2
  exit 1
fi

echo "[INFO] Using USB root: $USB_ROOT"

# 1) output -> ${USB_ROOT}/result
if [[ "$LINK_OUTPUT" == "1" ]]; then
  target="$USB_ROOT/result"
  if [[ -d "$target" ]]; then
    link_path "$target" output
  else
    echo "[WARN] Skip output link, not found: $target"
  fi
fi

# 2) data/<name> -> ${USB_ROOT}/dataset/<name>
if [[ "$LINK_DATASETS" == "1" ]]; then
  base="$USB_ROOT/dataset"
  if [[ -d "$base" ]]; then
    mkdir -p data
    shopt -s nullglob
    for d in "$base"/*; do
      name=$(basename "$d")
      link_path "$d" "data/${name}"
    done
    shopt -u nullglob
  else
    echo "[WARN] Skip dataset links, not found: $base"
  fi
fi

# 3) data/raw -> ${USB_ROOT}/data (オリジナルデータ)
if [[ "$LINK_RAW" == "1" ]]; then
  target="$USB_ROOT/data"
  if [[ -d "$target" ]]; then
    mkdir -p data
    link_path "$target" data/raw
  else
    echo "[WARN] Skip raw data link, not found: $target"
  fi
fi

echo "[DONE] Symlink setup complete."
