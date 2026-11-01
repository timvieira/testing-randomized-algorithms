## Open

Before publishing
- [ ] Remove the draft banner and set the date.

Optional
- [ ] Extract the tracing function into a small library (`tracing.py`) with tests.
  Should be pip-installable. Add `pyproject.toml`, `test_tracing.py`.
- [ ] Explain (or leave out) why SIS and SIR with 2 particles have exactly the
  same Var(Z-hat) on the toy model.

## Done

- [x] Estimator classes: SWOR is correlated but unbiased (V_eff < V); MCMC
  moves to biased but consistent.

- [x] Make SMC estimate a nontrivial Z (AWRS as the SMC proposal under a
  token-level constraint, so Z = P(constraint) < 1).
- [x] Reorder the AWRS variants so each testing strategy catches one more bug.
- [x] Remove duplicated code from rendered notebook cells.
- [x] Exact variance comparison (AWRS vs. naive).
- [x] Exploration orders: sampling and greedy, with a note on why true best-first
  needs replay and is fragile.
- [x] Interactive trace replay (D3), limitations section, takeaways recap.
- [x] Plots in Plotly, with plotting code in hidden helpers.
- [x] Credit Shi et al. (2020) for the tracing strategy, and cite WebPPL
  (Goodman & Stuhlmüller, 2014) for enumeration with continuations.
- [x] Match the prose's notation to the paper's pseudocode (x_c, psi_0).
- [x] CLAUDE.md: Pages serves from `master`; math is MathJax.
