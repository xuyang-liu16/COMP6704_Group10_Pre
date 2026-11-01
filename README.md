<div align="center">
<h1>COMP6704 Group 10 Presentation</h1>
<p><b>Convex Optimization for Machine Learning</b><br/>
L2-Regularized Logistic Regression</p>
<p>
  <strong>English</strong> | <a href="README_zh-CN.md">中文</a>
</p>
<p>
LIU Xuyang · WANG Zeqing · WU Aoqi · ZHAO Jixin<br/>
Department of Computing, The Hong Kong Polytechnic University<br/>
COMP6704 · Oral 3 November 2026 · 30 min + 10 min Q&A
</p>
</div>

We train the **same** L2-regularized logistic regression model with **four** convex solvers. Each member presents one method.

| Method | Speaker | Code |
| --- | --- | --- |
| Gradient Descent | WANG Zeqing | [`algorithms/gd.py`](algorithms/gd.py) |
| Nesterov accelerated gradient | LIU Xuyang | [`algorithms/nesterov.py`](algorithms/nesterov.py) |
| FISTA (proximal gradient) | WU Aoqi | [`algorithms/fista.py`](algorithms/fista.py) |
| ADMM | ZHAO Jixin | [`algorithms/admm.py`](algorithms/admm.py) |

## 📝 Contents

- [Problem](#-problem)
- [Algorithm locations](#-algorithm-locations)
- [Repository layout](#-repository-layout)
- [Quick start](#-quick-start)
- [Oral slides](#-oral-slides)
- [Speaking ownership](#-speaking-ownership)
- [Members](#-members)

## 🎯 Problem

Minimize the average logistic loss plus an L2 penalty:

\[
\min_w \; F(w) = f(w) + g(w), \quad
f(w)=\tfrac{1}{n}\sum_i \log\bigl(1+e^{-y_i x_i^\top w}\bigr),\quad
g(w)=\tfrac{\lambda}{2}\|w\|_2^2
\]

with \(\lambda=0.01\). Shared utilities live in [`problem.py`](problem.py), [`data.py`](data.py), and [`metrics.py`](metrics.py).

## 📂 Algorithm locations

| File | Method | Owner | What it implements |
| --- | --- | --- | --- |
| [`algorithms/gd.py`](algorithms/gd.py) | Gradient Descent | WANG Zeqing | Full-batch GD with Armijo backtracking; step based on \(1/L\) |
| [`algorithms/nesterov.py`](algorithms/nesterov.py) | Nesterov | LIU Xuyang | Look-ahead momentum; general-convex \(t\)-schedule; stop on \(\|\nabla F(w)\|\) |
| [`algorithms/fista.py`](algorithms/fista.py) | FISTA | WU Aoqi | Loss gradient + closed-form L2 proximal shrink; same momentum as Nesterov |
| [`algorithms/admm.py`](algorithms/admm.py) | ADMM | ZHAO Jixin | Split \(w/z\) with dual updates, residual checks, inner solves |

Do **not** cross-edit another member’s algorithm file. Ownership rules are also in [`AGENTS.md`](AGENTS.md).

Comparison driver and figures:

| Path | Role |
| --- | --- |
| [`experiments/run_compare.py`](experiments/run_compare.py) | Run all four solvers on synthetic / breast-cancer data |
| [`scripts/make_figures.py`](scripts/make_figures.py) | Plot objective vs iteration / wall time |
| [`results/`](results/) | Saved JSON tables and curves (do not invent numbers) |
| [`figures/`](figures/) | Export plots used in the deck |

## 🗂 Repository layout

```text
.
├── algorithms/          # Four solvers (one file each)
├── experiments/         # Comparison runner
├── figures/             # Plots for slides/report
├── results/             # summary.json, curves.json, tables
├── report/sections/     # Report markdown by section owner
├── slides/
│   ├── Group10_Oral.pptx              # Editable oral deck (LM fonts embedded)
│   ├── Group10_Oral_PolyU_Beamer.pdf  # Beamer visual reference
│   ├── Group10_Oral_PolyU_Beamer.tex
│   └── beamer/Group10/                # Beamer sources
├── SPEAKING_SCRIPTS_EN.md             # English talk scripts + timing
├── SPEAKING_SCRIPTS_BILINGUAL.md      # EN then 中文 notes
├── Group10_Oral.pptx                  # Root twin of slides/Group10_Oral.pptx
└── requirements.txt
```

## 🚀 Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Compare all four solvers (writes under results/)
python experiments/run_compare.py

# Rebuild figures
python scripts/make_figures.py
```

## 🎤 Oral slides

- Editable PPTX: [`slides/Group10_Oral.pptx`](slides/Group10_Oral.pptx) (32 visible + 2 hidden backups; Latin Modern embedded for WPS)
- Delivery notes: [`slides/PPTX_DELIVERY.md`](slides/PPTX_DELIVERY.md)
- Beamer PDF/TeX: [`slides/Group10_Oral_PolyU_Beamer.pdf`](slides/Group10_Oral_PolyU_Beamer.pdf)

## ⏱ Speaking ownership

| Part | Speaker | Slides | Clock |
| --- | --- | ---: | --- |
| Opening | LIU Xuyang | 1 | 00:00–00:25 |
| Background + shared formulation | WANG Zeqing | 2–6 | 00:25–04:20 |
| Four-method overview | LIU Xuyang | 7 | 04:20–05:00 |
| Gradient Descent | WANG Zeqing | 8–11 | 05:00–08:00 |
| Nesterov | LIU Xuyang | 12–15 | 08:00–12:00 |
| FISTA + equivalence | WU Aoqi | 16–19 | 12:00–16:00 |
| ADMM | ZHAO Jixin | 20–23 | 16:00–20:00 |
| Experiments + takeaways | ZHAO Jixin | 24–31 | 20:00–29:00 |
| Closing | LIU Xuyang | 32 | 29:00–30:00 |

Full scripts: [`SPEAKING_SCRIPTS_EN.md`](SPEAKING_SCRIPTS_EN.md) · [`SPEAKING_SCRIPTS_BILINGUAL.md`](SPEAKING_SCRIPTS_BILINGUAL.md)

## 👥 Members

| Name | NetID | Owns |
| --- | --- | --- |
| LIU Xuyang | 26123832r | `algorithms/nesterov.py`, opening/closing, formulation/conclusion |
| WANG Zeqing | 26029011r | `algorithms/gd.py`, shared setup, introduction |
| WU Aoqi | 26027031r | `algorithms/fista.py`, methods, experiment code helpers |
| ZHAO Jixin | 26009105r | `algorithms/admm.py`, results, experiment comparison slides |

## License / course use

Coursework for COMP6704 (PolyU). For teaching/demo use within the course.
