# Build the LeanNiche methods note

Install Git, uv, Pandoc 3.14 with its matching pandoc-crossref 0.3.26b release,
and XeLaTeX with the [required TeX packages](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/usage/output-formats.md).
The shared renderer uses Python 3.14 or later.

From a clean LeanNiche checkout, put a detached copy under the pinned
renderer. The registered copy carries its own source index.

```sh
project_root="$(pwd -P)"
work_directory="$(mktemp -d)"
renderer="$work_directory/template"
git clone https://github.com/docxology/template.git "$renderer"
git -C "$renderer" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$renderer/projects/working"
project_copy="$renderer/projects/working/lean_niche"
git clone --no-hardlinks "$project_root" "$project_copy"
git -C "$project_copy" checkout --detach "$(git -C "$project_root" rev-parse HEAD)"

cd "$renderer"
uv sync --locked --no-dev
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/lean_niche
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/lean_niche
```

Inspect `output/pdf/lean_niche_combined.pdf` and `output/web/index.html` in
`"$project_copy"` for the [ordered sections and bibliography](manuscript/README.md).
Stage 04 writes the validation report and rendered provenance in
`output/reports/`. The methods note contains text and citations. Its evidence
comes from Lean source and native tests, with no generated figure inputs.
