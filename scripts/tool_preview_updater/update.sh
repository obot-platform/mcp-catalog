#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/promptToUpdateToolPreviews.txt"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

is_allowed_yaml_path() {
  local path="$1"
  [[ "$path" =~ ^[^/]+\.yaml$ ]]
}

snapshot_workspace() {
  local prefix="$1"

  git -C "$REPO_ROOT" status --porcelain=v1 | LC_ALL=C sort > "$prefix.status"
  git -C "$REPO_ROOT" ls-files --others --exclude-standard | LC_ALL=C sort > "$prefix.untracked"

  : > "$prefix.hashes"
  while IFS= read -r path; do
    [[ -f "$REPO_ROOT/$path" ]] || continue
    printf '%s\t%s\n' "$(shasum -a 256 "$REPO_ROOT/$path" | awk '{print $1}')" "$path" >> "$prefix.hashes"
  done < <(git -C "$REPO_ROOT" ls-files | LC_ALL=C sort)
  LC_ALL=C sort -o "$prefix.hashes" "$prefix.hashes"
}

tracked_file_changes() {
  local before_hashes="$1"
  local after_hashes="$2"

  awk -F '\t' '
    NR == FNR {
      before[$2] = $1
      next
    }
    {
      seen[$2] = 1
      if (!($2 in before) || before[$2] != $1) {
        print $2
      }
    }
    END {
      for (path in before) {
        if (!(path in seen)) {
          print path
        }
      }
    }
  ' "$before_hashes" "$after_hashes" | LC_ALL=C sort -u
}

verify_workspace_changes() {
  local baseline_prefix="$1"
  local current_prefix="$TMP_DIR/current"
  local ok=1
  local removed_any=0

  snapshot_workspace "$current_prefix"

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if ! is_allowed_yaml_path "$path"; then
      printf 'Disallowed tracked file change: %s\n' "$path" >&2
      ok=0
    fi
  done < <(tracked_file_changes "$baseline_prefix.hashes" "$current_prefix.hashes")

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if ! grep -Fqx -- "$path" "$baseline_prefix.untracked"; then
      printf 'Removing new untracked file: %s\n' "$path" >&2
      rm -rf -- "$REPO_ROOT/$path"
      removed_any=1
    fi
  done < "$current_prefix.untracked"

  if [[ "$removed_any" -eq 1 ]]; then
    snapshot_workspace "$current_prefix"
  fi

  while IFS= read -r status_line; do
    local status path

    [[ -n "$status_line" ]] || continue
    if grep -Fqx -- "$status_line" "$baseline_prefix.status"; then
      continue
    fi

    status="${status_line:0:2}"
    path="${status_line:3}"

    if [[ "$status" == '??' || "$status" == *A* || "$status" == *D* || "$status" == *R* || "$status" == *C* || "$status" == *U* ]]; then
      printf 'Disallowed git status change: %s\n' "$status_line" >&2
      ok=0
      continue
    fi

    if ! is_allowed_yaml_path "$path"; then
      printf 'Disallowed git status change: %s\n' "$status_line" >&2
      ok=0
    fi
  done < "$current_prefix.status"

  [[ "$ok" -eq 1 ]]
}

require_command git
require_command opencode
require_command shasum
require_command awk

if [[ "$#" -gt 1 ]]; then
  printf 'Usage: %s [top-level-yaml-file]\n' "$0" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  printf 'Prompt file not found: %s\n' "$PROMPT_FILE" >&2
  exit 1
fi

if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'Not inside a git worktree: %s\n' "$REPO_ROOT" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BASELINE_PREFIX="$TMP_DIR/baseline"
snapshot_workspace "$BASELINE_PREFIX"

shopt -s nullglob
yaml_files=("$REPO_ROOT"/*.yaml)
shopt -u nullglob

if [[ "${#yaml_files[@]}" -eq 0 ]]; then
  printf 'No top-level YAML files found.\n'
  exit 0
fi

if [[ "$#" -eq 1 ]]; then
  requested_yaml="$1"

  if [[ "$requested_yaml" == */* ]]; then
    requested_yaml="$(basename "$requested_yaml")"
  fi

  if ! is_allowed_yaml_path "$requested_yaml"; then
    printf 'Argument must be a top-level YAML filename: %s\n' "$1" >&2
    exit 1
  fi

  requested_yaml_path="$REPO_ROOT/$requested_yaml"
  if [[ ! -f "$requested_yaml_path" ]]; then
    printf 'YAML file not found: %s\n' "$requested_yaml" >&2
    exit 1
  fi

  yaml_files=("$requested_yaml_path")
fi

printf 'Top-level YAML files:\n'
for yaml_file in "${yaml_files[@]}"; do
  printf ' - %s\n' "$(basename "$yaml_file")"
done

base_prompt="$(<"$PROMPT_FILE")"
processed_count=0

for yaml_file in "${yaml_files[@]}"; do
  if ! grep -q '^toolPreview:' "$yaml_file"; then
    continue
  fi

  processed_count=$((processed_count + 1))
  yaml_name="$(basename "$yaml_file")"
  run_prompt="$base_prompt"
  run_prompt+=$'\n\nONLY PROCESS ROOT-LEVEL FILE `'
  run_prompt+="$yaml_name"
  run_prompt+=$'`. Skip every other file. Do not create, delete, or rename any files.'

  printf '\nRunning opencode for %s\n' "$yaml_name"
  opencode run --dir "$REPO_ROOT" --model openai/gpt-5.4 --variant medium --agent build --dangerously-skip-permissions "$run_prompt"

  if ! verify_workspace_changes "$BASELINE_PREFIX"; then
    printf '\nWorkspace safety check failed after processing %s\n' "$yaml_name" >&2
    exit 1
  fi
done

printf '\nProcessed %d YAML file(s) with tool previews.\n' "$processed_count"
