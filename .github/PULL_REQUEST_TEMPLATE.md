<!--
Title format: <area>: <short imperative>
  Examples:
    infra: initialize pyproject.toml with deps, tool configs, entrypoints
    data: codingjepa.data.chunker (libcst-based top-level chunking)
    safety: stable refusal copy table (R001-R007)
-->

## Problem

<!-- 1-3 sentences: what is broken or missing, why it matters. Link the
relevant RFC or spec section if applicable. -->

## Solution

<!-- What this PR changes. Be specific about the public surface, schema
versions, or contract changes. Reference the linked issue's scope. -->

## Validation

<!-- Commands run and their outputs. Copy-paste actual output rather than
paraphrasing. At minimum: ruff / black / mypy / pytest for code-touching
PRs; markdown link-check for docs PRs. -->

```
$ make lint && make test
...
```

## Caveats

<!-- Known gaps, out-of-scope notes, and any follow-up issues. Be honest
about what is not done. -->

---

### Checklist

- [ ] PR title follows `<area>: <short imperative>`.
- [ ] Linked issue: `Closes #<n>` is in the body above.
- [ ] Relevant RFC / spec section linked under **Problem** or **Solution**.
- [ ] Tests added or extended for any code change (`docs/spec/07-testing-strategy.md`).
- [ ] `CHANGELOG.md` updated with an `[Unreleased]` entry **if** this PR
      touches `codingjepa/`, `data/schemas/`, or `docs/spec/02-public-api.md`.
- [ ] No tool/vendor branding in branch name, commits, or PR title
      (`docs/goal.md` §2.3).

Closes #<!-- issue number -->
