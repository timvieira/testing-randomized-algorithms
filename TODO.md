## Open

Before publishing
- [ ] Remove the draft banner and set the date.

## Done

- [x] `program_tracing` library in this repo (mirrors conditional-poisson-sampling):
  tests for the library and for the article's claims, ruff, coverage, CI, Makefile.

- [x] Investigated why SIS and SIR (2 particles) have identical Var(Z-hat): in
  the toy model, resampling only fires right before a deterministic final EOS
  step, and it preserves total weight, so Z-hat is unchanged on every run. A
  uniform-resampling bug passes a Z check but fails the per-string check.
  Decided to leave it out of the article (too much content).

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
