# CodingJEPA — MVP status

Current state: **documentation/spec package only.** No code, no model, no data.

## Completed
- [x] README (rebranded to CodingJEPA, Python focus)
- [x] PRD
- [x] system spec
- [x] research notes (incl. LeWorldModel mapping)
- [x] implementation plan
- [x] schedule
- [x] RFC stack (0001–0014)
- [x] candidate Python repositories list (`docs/data/CANDIDATE_REPOS.md`)

## Not started
- [ ] data audit and mirroring of the 10 source repos
- [ ] BPE tokenizer training
- [ ] chunking pipeline (`libcst`-based)
- [ ] baseline implementations (BM25, MLM-encoder, frozen CodeBERT)
- [ ] CodingJEPA encoder + predictor + projector implementation
- [ ] SIGReg implementation
- [ ] pretraining run
- [ ] intent fine-tuning run
- [ ] retrieval-rerank inference path
- [ ] FAISS index build
- [ ] demo CLI and web UI
- [ ] evaluation harness
- [ ] benchmark splits
- [ ] results memo
- [ ] paper draft

## Rule

Do not mark implementation milestones complete until the repository contains runnable or inspectable artifacts proving them. Spec-only completeness is not implementation completeness.

## Renaming note

This project was previously named **RustJEPA** and targeted Rust refactoring. As of 2026-05-15 it has been renamed to **CodingJEPA** and retargeted to Python. The Rust scoping documents are obsolete; the current spec stack is authoritative.
