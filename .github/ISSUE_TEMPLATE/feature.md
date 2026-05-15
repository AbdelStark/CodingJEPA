---
name: Feature / implementation issue
about: A unit of implementation work derived from the spec corpus.
title: "<area>: <imperative one-liner>"
labels: ["type:feature"]
assignees: []
---

## Context

<!-- 1-3 sentences. Why this matters and which spec section it derives
from. Include the RFC section if relevant. -->

## Scope

<!-- The smallest production-grade change that satisfies the spec. List
files / modules / schemas / CLI flags. -->

## Out of Scope

<!-- Explicit exclusions. Reference the issue numbers that own each
excluded item. -->

## Design Reference

- SPEC: [`docs/spec/<section>.md`](../blob/main/docs/spec/<section>.md)
- RFC: [`docs/rfcs/<rfc>.md`](../blob/main/docs/rfcs/<rfc>.md)

## Acceptance Criteria

- [ ] <criterion 1>
- [ ] <criterion 2>

## Dependencies

- Depends on: #<n>
- Blocks: #<m>

## Metadata

```
type: feature
area: infra | data | model | training | inference | eval | safety | demo | release | baselines
priority: p0 | p1 | p2
effort: s | m | l
spec: <rfc-id> | <spec-id>
milestone: v0.1 | v0.2 | v0.3 | v0.4 | v0.5 | v1.0
tracking: #<dashboard issue>
```
