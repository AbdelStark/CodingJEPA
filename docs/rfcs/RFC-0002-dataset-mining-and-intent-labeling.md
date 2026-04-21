# RFC-0002 — dataset mining and intent labeling

## Status
Draft

## Problem

Define repo selection, commit filtering, pair extraction, and intent labeling.

## Why this matters

If this decision stays fuzzy, the project will either optimize for the wrong target or bloat before the thesis is actually tested.

## Decisions to lock

- Repository quality criteria
- Commit classification heuristics
- Intent taxonomy and labels
- Gold subset construction plan

## Preferred v1 bias

Choose the smallest credible option that preserves demo speed and empirical honesty.

## Deferred items

- any move that broadens the project into a general platform
- any optimization that matters only after the first convincing demo exists
- any expansion in data/model size that does not materially change the first evaluation story

## Acceptance condition

This RFC is complete only when a builder could implement the next phase without guessing what the project is actually trying to prove.
