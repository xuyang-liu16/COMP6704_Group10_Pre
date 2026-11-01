# COMP6704 Group 10: oral presentation

Deliverables: `slides/Group10_Oral.pptx` and the byte-identical root copy
`Group10_Oral.pptx`.

The refined presentation has **32 visible talk slides and 2 hidden Q&A
backups**. The talk lasts 30 minutes. Each member presents one solver.
The 35-page Group 17 reference informed the page count, section labels,
algorithm sequences, whitespace and native result-table layouts.

## Order and ownership

| Part | Speaker | Slides | Duration | Clock |
| --- | --- | --- | --- | --- |
| Opening | LIU Xuyang | 1 | 25s | 00:00-00:25 |
| Background and shared formulation | WANG Zeqing | 2-6 | 3m 55s | 00:25-04:20 |
| Four-method overview | LIU Xuyang | 7 | 40s | 04:20-05:00 |
| Gradient Descent | WANG Zeqing | 8-11 | 3m | 05:00-08:00 |
| Nesterov | LIU Xuyang | 12-15 | 4m | 08:00-12:00 |
| FISTA and equivalence | WU Aoqi | 16-19 | 4m | 12:00-16:00 |
| ADMM | ZHAO Jixin | 20-23 | 4m | 16:00-20:00 |
| Experiments and comparison | ZHAO Jixin | 24-31 | 9m | 20:00-29:00 |
| Closing | LIU Xuyang | 32 | 1m | 29:00-30:00 |

WANG owns the objective, gradient, uniqueness and common stopping certificate.
WU owns the Nesterov-FISTA equivalence. ZHAO owns every experiment,
comparison and limitation slide, plus the ADMM residual backup on slide 33.
LIU owns the references backup on slide 34. Both backups are hidden and
carry no planned talk time.

The opening line remains:

> We use four methods, GD, Nesterov, FISTA and ADMM, to train the same model.
> Each of us presents one.

Each algorithm has four slides. FISTA's allocation is 40s, 50s, 50s and
100s, leaving time to explain the equivalence derivation. The visible-slide
durations sum to exactly 1,800 seconds. The remaining allocations and exact
clock windows appear in both rehearsal scripts.

## Notes and scripts

- `SPEAKING_SCRIPTS_EN.md`: the updated timing table and 34 English scripts.
- `SPEAKING_SCRIPTS_BILINGUAL.md`: the same English scripts followed by
  `中文：` and their complete Chinese translations.
- Each PPTX note gives the speaker, topic, clock window and duration.
  Its bilingual content matches the corresponding rehearsal script.
- Speaker labels use English names on every slide. Chinese remains in
  notes and the original bilingual university artwork.

The longer deck splits existing ideas across more pages. It retains the
same objective, assumptions, solver settings, comparison and limitations.
No experiments were rerun.

## Design and fonts

The cover preserves the original crimson band, logos, watermark and editable
Latin Modern title. A small opening-speaker label identifies LIU Xuyang.
Body slides use the PolyU logo, a thin maroon rule, a section label, a large
LM Roman title and an English speaker footer. Equations, text, process
diagrams and nine tables remain editable.

The three original plot images retain their source bytes. The two result
tables reproduce the saved JSON values at the existing display precision.
The separate 1,000-update ADMM check stays outside the common-budget results.

All nine embedded font parts are byte-identical to the previous LM deck:

- Latin Modern Roman: regular, bold, italic and bold italic.
- Latin Modern Sans: regular, bold, italic and bold italic.
- Latin Modern Math: regular.

The font parts remain under `ppt/fonts/`. The presentation declares
`embedTrueTypeFonts=1` and `saveSubsetFonts=0`, and retains the original
embedded family and style mappings. Native text uses only Latin Modern
Roman, Sans and Math. Mathematical subscripts and superscripts use native
baseline formatting. No Times-only substitution was introduced.

## Verification

The refinement checks cover:

- 34 slides, with slides 1-32 visible and 33-34 hidden.
- Correct per-solver speaker labels and 34 bilingual notes.
- Exactly 1,800 seconds of planned talk time.
- Nine editable native tables on slides 7, 11, 15, 18, 23, 24, 25, 28 and 30.
- Both result tables checked against `results/summary.json` and
  `results/breast_cancer/summary.json`.
- Byte-identical embedded fonts, preserved plot data and unchanged
  algorithm, experiment and result files.
- Artifact Tool import of all 34 slides and successful package, font-policy
  and slide-geometry validation.
- LibreOffice render of all 34 pages with the existing embedded-only
  font configuration. The PDF uses Latin Modern Roman, Sans and Math.
- No detected text-line collisions or text outside the page. Visual review
  covers every page, including the cover, algorithm equations and results.
- Identical SHA-256 values for the two delivered PPTX copies.

**Application boundary:** LibreOffice rendering and package checks are
verified. This revision has not undergone a WPS or Microsoft PowerPoint
open/edit/save round-trip. The unchanged cover and embedded font bytes
preserve the supplied WPS-oriented design, but do not establish behavior
on every WPS version or device.

## Build records

Current build and QA records are under `.pptx-build/g17-refined/`:

- `plan.json`: the slide order, ownership, timing and bilingual notes.
- `preservation.json`: source font hashes and preserved plot mappings.
- `protected-hashes.json`: algorithm, experiment and result file hashes.
- `validation.json`: Artifact Tool and structural validation receipt.
- `qa.json`: source-data, notes, ownership and rendered-text checks.
- `render/candidate.pdf` and `render/pages/slide-01.png` through
  `slide-34.png`: the LibreOffice render, including hidden backups.
- `delivery.json`: final file hashes and visual-review record.

Build entry points are `.pptx-build/refine_g17.mjs`,
`refine_g17_package.py`, `refine_g17_finalize.mjs` and `refine_g17_qa.py`.
The builder uses Artifact Tool for editable content, then the packaging
pass preserves the original cover, plot images and embedded font parts.
Historical `per_solver_*`, `wps_*` and `lm_*` files describe earlier
revisions. The Beamer PDF remains an unchanged visual reference.

No member-owned algorithm, report, slide markdown or experiment source was
modified. No system fonts were installed or removed.
