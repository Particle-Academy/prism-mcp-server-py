# Working on prism-mcp-server-py

Read `../prism-parity/docs/AGENTS.md` and the authorized envelope spec
`../../.ai/plans/prism-mcp-server-first-cut.md`. Phases 1 and 2 are authorized;
phase 3 (SDK conformance), remotes, publication and tags are not. Shared runtime
types come from prism_mcp. Stage explicit paths; no AI commit attribution.

Run `python -m pytest`, `python -m mypy --strict src tests`, `python -m ruff check .`,
`python -m ruff format --check .` and `git diff --check`. Redirect each gate to a
file and echo its exit status; do not pipe a gate. Before a commit, read the
envelope `.claude/skills/security-review/SKILL.md`.
