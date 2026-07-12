# Manuscript — Certifying Regime Detectors Before Use

LaTeX source, compiled PDF, figures, and provenance for the paper
**"Certifying Regime Detectors Before Use"** (M5 Research).

## Contents

| File | Purpose |
|---|---|
| `main.tex` | LaTeX source (elsarticle, `3p`). |
| `main.pdf` | Compiled manuscript. |
| `references.bib` | Bibliography. |
| `main.bbl` | Pre-compiled bibliography (lets `pdflatex` build without a separate `bibtex` run). |
| `figures/` | The four figures used by the manuscript. |
| `REPRODUCTION.md` | Exact manuscript-build and artifact-regeneration commands. |
| `CLAIMS.md` | Allowed claims, forbidden claims, and wording guardrails. |
| `CLAIM_TRACEABILITY.md` | Claim → artifact → status map for every manuscript claim. |
| `ARTIFACT_LEDGER.md` | Verified ledger of every evidence artifact behind the paper. |
| `HARMONIZED_BENCHMARK_PROTOCOL.md` | Protocol for the three-family detector benchmark. |

## Build

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`main.bbl` is included, so a single `pdflatex main.tex` also builds the PDF without
running `bibtex`. See [`REPRODUCTION.md`](REPRODUCTION.md) for regenerating the
underlying evidence artifacts.

## Legacy naming

The protocol was developed under the working name **GCDE** ("gauge-calibrated
detector exclusion"); some frozen artifact paths and keys retain that vocabulary.
The manuscript's Reproducibility appendix documents the mapping. See the repository
root [`README.md`](../README.md) for the full replication package.
