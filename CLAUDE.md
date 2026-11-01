# Testing Randomized Algorithms

Blog post about ideas for testing randomized algorithms—program tracing, importance sampling diagnostics, and related techniques.

## Project structure

- `content/testing-randomized-algorithms.md` — main article (Markdown with YAML front matter, inline `<style>`, MathJax math)
- `content/testing-randomized-algorithms.ipynb` — Jupyter notebook; cells are pulled into the article via `{% notebook %}` directives
- `content/images/` — figures referenced by the article
- `content/widgets/` — the interactive trace replay (`trace-replay.js`) and its data (`trace-data.js`, written by the notebook's last cell)
- `program_tracing/` — the tracer as a small library; the notebook imports it, and the article prints its source
- `tests/test_tracing.py` — library tests on programs with known distributions
- `tests/test_article.py` — tests of the claims made in the article
- `pyproject.toml` — package metadata, dev dependencies, ruff and pytest config, and the blog build config (`[tool.blog]`)
- `Makefile` — `make test`, `lint`, `coverage`, `notebook`, `dev`, `deploy`, `install`
- `.github/workflows/tests.yml` — CI: ruff, pytest with coverage, and a full run of the notebook
- `docs/` — built output served by GitHub Pages (do not edit directly)

## Build system

Uses the centralized `blog` CLI (`~/.local/bin/blog`).

```bash
blog dev          # build + watch + serve (proxies /blog/ to the blog repo so /blog/css/blog.css resolves)
blog build        # one-shot build
blog deploy       # build to docs/, commit, and push
```

## Article ↔ tests

Every claim the article makes about results (which variants pass, M* values, variance ratios, trace counts) has a test in `tests/test_article.py`, and the tracer code printed in the article must be exactly the library's source (also tested).

- When changing a claim in the article, update its test in the same commit.
- When changing `program_tracing/tracer.py`, update the article's code blocks to match.
- If a test fails, the article is wrong until proven otherwise; don't weaken the test to make it pass.
- Run `make test` and `make lint` before committing. After changing the notebook's code, run `make notebook` so the saved outputs and widget data stay current.

## Deployment

- GitHub Pages serves from `docs/` on the `master` branch (satellite mode — article is the root index).
- Site URL: `https://timvieira.github.io/testing-randomized-algorithms/`
- `blog deploy` handles the full build-commit-push cycle.

## Style and conventions

- **Font**: EB Garamond (loaded from the blog's `blog.css`).
- **Writing**: Understated tone, no dorky capitalization, no cliche quotes. American English.
- **Punctuation**: Tight em-dashes (word—word), not spaced en-dashes.
- **Math**: MathJax with `$...$` (inline) and `$$...$$` (display). Use `\widehat{}` not `\hat{}`. Use `^\top` not `^T`.
