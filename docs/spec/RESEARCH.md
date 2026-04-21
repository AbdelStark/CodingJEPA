# RustJEPA — research notes

## Project domain

code representation learning and intent-conditioned refactoring

## Central research question

A JEPA-style model trained on before/after Rust refactor pairs can learn semantic invariants and predict intent-conditioned target representations that are better suited to narrow refactoring tasks than token-level generation alone.

## Data sources under consideration

- Filtered Rust commit pairs from stable OSS repositories
- Intent-labeled subset for narrow refactor tasks
- Optional synthetic augmentation for small controlled transformations

## Questions to answer before implementation grows

1. What is the smallest task framing that still tests the thesis honestly?
2. What is the strongest cheap baseline?
3. What nuisance or confounders matter most in practice?
4. What would count as a fake win here?
5. What result would actually persuade a skeptical technical reader?

## Research anti-patterns

- overclaiming beyond the chosen task
- broadening the problem until the evaluation becomes vague
- substituting visual polish for empirical clarity
- comparing only against weak baselines
- letting the demo narrative outrun the measured result
