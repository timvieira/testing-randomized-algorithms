- [ ] Extract the tracing function into a small library (`tracing.py`) with tests.
  Should be pip-installable. Add `pyproject.toml`, `test_tracing.py`.

- [ ] Make SIS/SIR estimate a nontrivial Z (e.g., Z=2). Requires changing the
  weight update to use unnormalized target factors divided by normalized proposal,
  so the product of incremental weights telescopes correctly. Currently Z=1
  because both target and proposal are normalized.
