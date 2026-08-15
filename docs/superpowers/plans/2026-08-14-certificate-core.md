# Certificate Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a detector-independent certificate-v1 schema, disposition classifier, geometric lineage, and create-only runner that the BTC successor and EUR/USD certificates will both consume.

**Architecture:** Pydantic v2 models in `src/certificate/` own validation and hashing. Gate code returns `GateResult` only. The runner classifies completeness and disposition after every required gate has a recorded state. Existing artifacts are never overwritten.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest

---

## File Map

- Create: `src/certificate/errors.py`
- Create: `src/certificate/models.py`
- Create: `src/certificate/serialize.py`
- Create: `src/certificate/disposition.py`
- Create: `src/certificate/lineage.py`
- Create: `src/certificate/preregistration.py`
- Create: `src/certificate/runner.py`
- Create: `src/certificate/__init__.py`
- Create: `scripts/wp1/run_certificate.py`
- Create: `schemas/certificate-v1.json`
- Create: `tests/wp1/test_certificate_disposition.py`
- Create: `tests/wp1/test_certificate_schema.py`
- Create: `tests/wp1/test_certificate_lineage.py`
- Create: `tests/wp1/test_certificate_serialize.py`
- Create: `tests/wp1/test_run_certificate.py`
- Modify: `pyproject.toml`, `requirements.txt`

### Task 1: Disposition classifier

**Files:**
- Create: `tests/wp1/test_certificate_disposition.py`
- Create: `src/certificate/disposition.py`

- [ ] Write failing tests for missing-gate `incomplete` override, instrument/target precedence, and failure-profile retention
- [ ] Run tests and confirm they fail
- [ ] Implement `classify_disposition`
- [ ] Run tests and confirm they pass

### Task 2: Schema, hashing, lineage

**Files:**
- Create: `tests/wp1/test_certificate_schema.py`
- Create: `tests/wp1/test_certificate_serialize.py`
- Create: `tests/wp1/test_certificate_lineage.py`
- Create: `src/certificate/models.py`
- Create: `src/certificate/serialize.py`
- Create: `src/certificate/lineage.py`

- [ ] Write failing tests for extra fields, non-finite estimands, duplicate IDs, scope mutation, geometric alpha, and metadata patches
- [ ] Implement models, canonical JSON, SHA-256 spec hashing, and revision-tree validation
- [ ] Export `schemas/certificate-v1.json`

### Task 3: Runner

**Files:**
- Create: `tests/wp1/test_run_certificate.py`
- Create: `src/certificate/runner.py`
- Create: `src/certificate/preregistration.py`
- Create: `scripts/wp1/run_certificate.py`

- [ ] Write failing tests for admitted, withheld, incomplete, and invalid fixtures; overwrite refusal; dirty/missing preregistration
- [ ] Implement `python -m scripts.wp1.run_certificate`
- [ ] Exit 0 for a completed scientific certificate, including withheld; nonzero for invalid execution
