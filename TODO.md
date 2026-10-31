- [ ] Extract the tracing function into a small library (`tracing.py`) with tests.
  Should be pip-installable. Add `pyproject.toml`, `test_tracing.py`.

- [x] Make SMC estimate a nontrivial Z. Done by making AWRS the SMC proposal
  under a token-level constraint, so Z = P(constraint) < 1.

- [x] Reorder the AWRS variants so each testing strategy catches one more bug
  (B: single run, C: multiple runs, D: tracing only).

- [x] Remove duplicated code from rendered notebook cells.
