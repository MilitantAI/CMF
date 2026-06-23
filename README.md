# Cogenetic Minkowski Functional (CMF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20808466.svg)](https://doi.org/10.5281/zenodo.20808466)

Reference paper and Python implementation. [Cogenesis](https://doi.org/10.5281/zenodo.20539999) is a residual-aware framework; CMF is its Minkowski-stage functional.

**Read first:** [cmf.pdf](cmf.pdf) · [Zenodo record](https://doi.org/10.5281/zenodo.20808466)

## Quick summary

Standard practice treats Minkowski `(M, η)` as primitive spacetime---reifying a calculus accounting hack---and bolts environment and process on as external inputs. CMF instead:

- Treats `(M, η)` as **projection calculus** (utilitarian simplification), distinct from **process calculus** — aligned with [Cogenesis](https://doi.org/10.5281/zenodo.20539999)
- Keeps the machinery; rejects fundamentalising time as a fourth dimension
- Declares **C** (environmental density gradient along the observer throughline)
- Uses process time **t** indexing resolution **R(t)** = (ℓ, τ, Λ, ε, Σ)
- **Minkowski stipulates** — fixes `|r|=0`, `g=η`, and a functional with no R-map
- **CMF determines** — computes residual, metric correction, and `O(C,R(t))` at each t
- **CMF predicts** — refinement behaviour as `R(t)` sharpens
- Recovers flat `η` when residual vanishes — when the simplification suffices, not because spacetime is true

## Build the PDF

```bash
pdflatex cmf.tex
pdflatex cmf.tex
```

Or: `make pdf` (requires TeX Live or MiKTeX)

## Run the demo

```bash
pip install -r requirements.txt
python demo.py
python demo.py --open   # optional: open comparison figure after run
```

Regenerates figures in `output/` embedded by the paper.

## Related work

- [Cogenesis](https://doi.org/10.5281/zenodo.20539999) — process-first, residual-aware ontology
- [One-Metric Quantum Gravity](https://doi.org/10.5281/zenodo.17731104) — shared causal cone and effective metric across sectors

## Repository layout

```
Minkowski/
  cmf.tex            LaTeX source
  cmf.pdf            Compiled paper
  model.py           CMF core implementation
  demo.py            Satellite point-detector demonstration
  requirements.txt   Python dependencies
  Makefile           Build cmf.pdf
  output/            Figures for the paper
  view.html          Local figure gallery
  LICENSE            CC BY 4.0
```

## Citation

```bibtex
@software{petrik2026cmf,
  author    = {Petrik, Morgan},
  title     = {Cogenetic Minkowski Functional (CMF)},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20808466},
  url       = {https://doi.org/10.5281/zenodo.20808466}
}
```

Plain text:

> Morgan Petrik. *Cogenetic Minkowski Functional (CMF).* Militant.AI, 23 June 2026. Version 1.0.0. https://doi.org/10.5281/zenodo.20808466

## License

[CC BY 4.0](LICENSE) — Morgan Petrik, Militant.AI.
