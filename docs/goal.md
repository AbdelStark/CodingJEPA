# Burn The Backlog: Autonomous Issue Resolution

Implement every open, unassigned, actionable GitHub issue in this repository to production-grade standards. Work **sequentially**, **one issue per PR**, and continue until the queue is exhausted.

## 0. Repository Reconnaissance

Before any edits, inspect:
- Repo state: current branch, working tree, remotes, `gh auth status`.
- Issue surface: open issues, labels, milestones, assignees, linked PRs, CI status.
- Use `gh issue list` and filter out:
  - Assigned issues
  - `duplicate` / `wontfix` / `blocked`
  - Issues with an active (non-stale) PR

## 1. Execution Plan

Produce an ordered queue **before editing code**. Order by:
1. Dependency (foundations before dependents)
2. Risk (de-risk contracts and quality gates first)
3. Project value (vertical slices over broad partial changes)

Rules:
- Do not skip hard issues for cosmetic ones.
- Separate design/RFC issues from implementation issues.
- For blocked issues: post a GitHub comment with the blocker, then move on.

## 2. Per-Issue Loop (Sequential — One Issue at a Time)

Do **not** start issue N+1 until issue N's PR is opened and the issue is commented/closed appropriately.

For each issue:

1. **Read** issue body, labels, linked docs, referenced files, related issues/PRs.
2. **Inspect** relevant code and tests before deciding implementation.
3. **Branch**: `git checkout -b <descriptive-name>` from latest default branch. No tool/vendor branding.
4. **Implement** the smallest complete production-grade change satisfying acceptance criteria.
5. **Test**: add or update focused tests for bug fixes, contracts, CLI behavior, provider behavior, artifact schemas, docs rules, and documented failure modes.
6. **Docs**: update docs, changelog, generated docs, examples, and agent/context files **only when public behavior changes**.
7. **Validate**: run the strongest relevant local validation (targeted tests → docs/build checks if public surface changed → broader gates if warranted).
8. **Commit**: one clean human commit message per logical issue. Reference issue number in commit/PR body. No tool branding.
9. **Open one PR for this issue only.** PR body must contain:
   - Problem
   - Solution
   - Validation (commands run + results)
   - Caveats
   - `Closes #<n>`
10. **Comment** on the issue with implementation summary, commit/PR link, validation output.
11. **Close** the issue only when implementation fully satisfies acceptance criteria and validation passed.
12. **Return to default branch**, pull, and proceed to the next issue.

## 3. Hard Constraints

- **One issue = one branch = one PR.** Never bundle issues.
- Fetch before pushing. Never reset or revert work you did not author.
- Preserve existing architecture, optional-runtime boundaries, security boundaries, repo conventions.
- Do not weaken: coverage gates, CI, validation, public API contracts, provider capability truthfulness, security redaction.
- Optional dependencies remain host-owned unless the issue explicitly changes that policy.
- Provider capabilities must remain truthful — never advertise unsupported operations.
- JSON artifacts: schema-versioned, JSON-native, finite, sanitized, internally coherent.
- Security-sensitive logs, provider events, manifests, reports, and issue bundles: redacted and safe by default.
- Prefer explicit failures over silent coercion.
- No broad abstractions unless they remove real complexity or match existing patterns.

## 4. Communication Quality

Public docs, PR descriptions, and issue comments are maintainer-facing. Be direct, precise, no hype, no marketing language.

## 5. Validation Strategy

1. Reproduce or characterize the issue's contract first.
2. Run targeted tests for the changed surface.
3. Run docs/build/generated checks if docs or public surfaces changed.
4. Run broader gates after a batch or before publishing.
5. If full validation is unavailable, state exactly what was run, what was skipped, and why.

## 6. Final Report

After the queue is exhausted, emit a concise report:

- **Completed**: issue → PR → validation summary
- **Skipped/blocked**: issue → reason → evidence
- **Out of scope**: issue → justification
- **Residual risks**: anything a maintainer should know

Continue autonomously until every open, unassigned, actionable issue is implemented, blocked with evidence, or explicitly out of scope.
