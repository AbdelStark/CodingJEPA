# CodingJEPA — research notes

## Project domain

Self-supervised code representation learning and intent-conditioned latent prediction for Python.

## Central research question

Does a JEPA-style training recipe — next-embedding prediction + a distributional regularizer over the embedding space — produce Python code representations that transfer to narrow refactoring and code-understanding tasks better than parameter- and FLOP-matched token-level baselines?

## Hypothesis

**H1.** Predicting the embedding of the next code chunk rather than its tokens removes pressure on the model to reconstruct surface-level nuisance variables (formatting, naming, comments), and concentrates capacity on semantic features (control flow, API shape, side-effect surface).

**H2.** Conditioning the predictor on a discrete refactor intent ("the action") produces a one-shot mapping from `(source_latent, intent) → target_latent` that is useful for retrieval-rerank-based refactoring without needing a generative decoder.

**H3.** The LeWorldModel two-term loss (prediction MSE + SIGReg) transfers to the code modality without requiring an EMA target encoder, multi-block masking schedules (a la I-JEPA), or teacher–student decoupling (a la V-JEPA). I.e., a single network suffices.

## Prior art and influences

### JEPA family
- **I-JEPA** (Assran et al., 2023): masked-block latent prediction on images; introduced the JEPA training framing for representations.
- **V-JEPA / V-JEPA 2** (Bardes et al., 2024–2025; LeCun et al.): extended to video; uses an EMA target encoder. Demonstrated that EMA can be brittle and hyperparameter-heavy.
- **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026): showed that end-to-end JEPA training is stable without EMA, with a single regularizer (SIGReg). Action-conditioned world model on pixels. **This is the architectural and training-recipe substrate for CodingJEPA.**

### Code representation learning
- **CodeBERT** (Feng et al., 2020) — MLM + replaced-token detection on code.
- **GraphCodeBERT** (Guo et al., 2021) — adds data-flow edges.
- **UniXcoder** (Guo et al., 2022) — unified cross-modal encoder.
- **CodeT5/CodeT5+** (Wang et al., 2021/2023) — encoder–decoder for code.
- **StarCoder** / **CodeLlama** — large decoder-only models; not a like-for-like comparison.

### Latent prediction for code
- **CodeRetriever** (Li et al., 2022) — contrastive retrieval over function-comment pairs.
- **SantaCoder / Replit-code** — decoder-only generation.
- We are not aware of prior work that applies a JEPA-style next-embedding objective with a distributional regularizer to code. This is our slot.

### Refactor mining
- **Refactoring Miner** (Tsantalis et al., 2018–) — Java-focused but the heuristics carry over.
- **PyDriller** — commit walking for Python.
- **Rope** / **LibCST** — Python AST/CST tooling.

## Why Python (not Rust)

- Larger and more diverse high-quality corpus of refactor pairs (decades of stdlib history, ML frameworks, web frameworks).
- Dynamic typing makes the semantic-vs-syntactic gap more pronounced — a better stress test for the thesis.
- Existing high-quality CST tooling (`libcst`, `ast`, `parso`) makes the chunking pipeline tractable.
- Easier to construct execution-preservation checks via `pytest` on real test suites.
- The 10-repo source list (see `docs/data/CANDIDATE_REPOS.md`) covers language internals, web, data, ML, type analysis, and tooling — broad enough to stress generality, narrow enough to control.

## Mapping LeWorldModel concepts to code

| LeWorldModel | CodingJEPA |
|---|---|
| Pixel frame | Code chunk (function / class / top-level block) |
| Frame sequence in time | Sequence of chunks within a file or across files (import topology) |
| ViT image encoder | Transformer encoder over BPE tokens |
| Action (`act_emb`) | Intent label embedding (`[I_*]`) — or `[I_NONE]` for unconditional pretraining |
| `ARPredictor` over history | Same: predicts next-chunk latent from `H` prior chunk latents + intent |
| EMA target | **Removed.** Target is the same encoder with stop-gradient. |
| SIGReg over embedding distribution | Same. Sliced isotropic Gaussian regularizer. |
| Rollout for planning | Beam-rollout for multi-step refactor chains (deferred to v2) |
| `get_cost(actions)` for MPC | `score(intent)` for retrieval ranking |

## Questions to answer before implementation grows

1. **Chunk granularity.** Per-function chunks vs. per-statement-block chunks vs. mixed? Locked in RFC-0012 (functions + class bodies, capped at 512 BPE tokens; longer items truncated with a `[TRUNC]` token).
2. **Intent vocabulary.** Eight intents (RFC-0004) — chosen for high-recall heuristic labeling and clear acceptance rules.
3. **Sequence construction.** File-order vs. call-graph topological order? **File-order for v1**, call-graph deferred to v2.
4. **Are refactor pairs cleanly identifiable?** Use a small, conservative heuristic set per intent + manual gold subset for evaluation. RFC-0002.
5. **What is the strongest cheap baseline?** Frozen CodeBERT retrieval. RFC-0005.
6. **What confounders matter most?** Formatting drift, identifier renames, docstring edits, import reordering. Mitigated by RFC-0012's normalization and RFC-0010's invariance probes.
7. **What would count as a fake win?** (a) Beating BM25 only because of corpus contamination. (b) High retrieval @ k against a tiny pool. (c) Cherry-picked intents.
8. **What would persuade a skeptical reader?** A clean ablation table showing (i) JEPA loss beats MLM at matched compute, (ii) SIGReg matters, (iii) intent conditioning matters, (iv) execution-preservation holds on real tests.

## Research anti-patterns we explicitly avoid

- Overclaiming beyond the chosen task family.
- Broadening the problem until evaluation becomes vague.
- Substituting visual polish for empirical clarity.
- Comparing only against weak baselines.
- Letting the demo narrative outrun the measured result.
- Reporting only the best seed.

## Open questions deferred to v2

- Multi-step refactor chains (planning in latent space).
- Cross-file refactors (move-function-across-modules).
- Multi-language extension (TypeScript, Go).
- Generative decoder head trained on top of frozen latents.
