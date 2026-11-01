# Testing Randomized Algorithms

Blog post about ideas for testing randomized algorithms—program tracing, importance sampling diagnostics, and related techniques.

## Project structure

- `content/testing-randomized-algorithms.md` — main article (Markdown with YAML front matter, inline `<style>`, MathJax math)
- `content/testing-randomized-algorithms.ipynb` — Jupyter notebook; cells are pulled into the article via `{% notebook %}` directives
- `content/images/` — figures referenced by the article
- `blog.toml` — blog build config (`static = ["images"]`)
- `docs/` — built output served by GitHub Pages (do not edit directly)
- `TODO.md` — task tracking

## Build system

Uses the centralized `blog` CLI (`~/.local/bin/blog`).

```bash
blog dev          # build + watch + serve (proxies /blog/ to the blog repo so /blog/css/blog.css resolves)
blog build        # one-shot build
blog deploy       # build to docs/, commit, and push
```

## Deployment

- GitHub Pages serves from `docs/` on the `master` branch (satellite mode — article is the root index).
- Site URL: `https://timvieira.github.io/testing-randomized-algorithms/`
- `blog deploy` handles the full build-commit-push cycle.

## Style and conventions

- **Font**: EB Garamond (loaded from the blog's `blog.css`).
- **Writing**: Understated tone, no dorky capitalization, no cliche quotes. American English.
- **Punctuation**: Tight em-dashes (word—word), not spaced en-dashes.
- **Math**: MathJax with `$...$` (inline) and `$$...$$` (display). Use `\widehat{}` not `\hat{}`. Use `^\top` not `^T`.
- **TODO tracking**: Update `TODO.md` after completing each task.
