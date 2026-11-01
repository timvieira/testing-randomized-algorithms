## Open

Accuracy
- [ ] "Correlated but unbiased (e.g., MCMC)": MCMC isn't unbiased in general.
- [ ] "V_eff > V" isn't guaranteed (negative autocorrelation can make it smaller).
- [ ] Check that Shi et al. (2020) is the right citation for unbiased estimates
  from early-stopped sampling without replacement; Kool et al. (2019) may fit better.

Polish
- [ ] Move matplotlib boilerplate into hidden helper cells so plot cells are short.
- [ ] Align notation with the paper's pseudocode (text uses x* and psi; image uses
  x_c and psi_0).
- [ ] CLAUDE.md says GitHub Pages serves from `main`; the branch is `master`.
- [ ] Before publishing: remove the draft banner and set the date.

Optional
- [ ] Extract the tracing function into a small library (`tracing.py`) with tests.
  Should be pip-installable. Add `pyproject.toml`, `test_tracing.py`.

## Done

- [x] Make SMC estimate a nontrivial Z (AWRS as the SMC proposal under a
  token-level constraint, so Z = P(constraint) < 1).
- [x] Reorder the AWRS variants so each testing strategy catches one more bug.
- [x] Remove duplicated code from rendered notebook cells.
- [x] Exact variance comparison (AWRS vs. naive).
- [x] Exploration orders: sampling and greedy, with a note on why true best-first
  needs replay and is fragile.
- [x] Interactive trace replay (D3), limitations section, takeaways recap.
