# .github/workflows/ — CI Workflows

GitHub Actions workflows for the LeanNiche repository.

## Contents
- `ci.yml` — main CI: build + test with Lean verification (push/PR to main)
- `lean_verification.yml` — Lean proof verification workflow
- `lean_action_ci.yml`, `lean-dashboard.yml`, `lean-health-monitor.yml`, `lean-realness-check.yml`, `lean-verification-summary.yml`, `scheduled-lean-verification.yml` — Lean-focused CI dashboards, health monitoring, and scheduled verification runs

All files are standard GitHub Actions YAML; triggers and jobs can be read directly from each file.
