#!/usr/bin/env bash
# Write local source metrics while keeping authored documentation intact.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

destination="outputs/documentation/code_metrics.md"
mkdir -p "$(dirname "$destination")"

{
    printf '# LeanNiche source metrics\n\n'
    printf '| Files | Count | Lines |\n| --- | ---: | ---: |\n'
    for extension in lean py; do
        files="$(find src -type f -name "*.$extension" -print | wc -l)"
        lines="$(find src -type f -name "*.$extension" -exec cat {} + | wc -l)"
        printf '| `*.%s` | %d | %d |\n' "$extension" "$files" "$lines"
    done
} > "$destination"

printf '%s\n' "$destination"
