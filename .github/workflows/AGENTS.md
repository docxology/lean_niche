# GitHub Actions Workflows

**Layout**: ci.yml, lean-dashboard.yml, lean-health-monitor.yml, lean-realness-check.yml, lean-verification-summary.yml, lean-verification.yml, lean_action_ci.yml, scheduled-lean-verification.yml
**Primary CI**: ci.yml for main build/test pipeline
**Lean verification**: Multiple specialized workflows for proof checking
**Monitoring**: Health and dashboard workflows
**Scheduling**: Automated verification runs

Verify: `gh workflow list` to check status
See parent README.md for workflow documentation.
