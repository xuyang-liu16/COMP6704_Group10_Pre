# COMP6704 Group 10 oral delivery

## Final files

- Presentation PDF: `slides/Group10_Oral_PolyU_Beamer.pdf`
- Beamer source: `slides/beamer/Group10/main.tex`
- Build output: `slides/beamer/Group10/main.pdf`
- Full spoken script: `SPEAKING_SCRIPTS_EN.md`
- Delivery guide: `slides/DELIVERY.md`

Paths are relative to the repository root.
The PDF has **26 pages: 24 talk pages and 2 Q&A backup pages**.
The source keeps the PolyU theme, formulas, figures and numerical tables.
All names use English characters. Every talk frame has a short `\note{...}`
with 2-5 spoken sentences. These notes stay in the source and do not add
pages to the audience PDF. The Markdown script contains the full talk.

## Timing and word counts

The schedule is exactly **30 minutes**, followed by questions.
Counts include only spoken text on pages 1-24. They exclude headings,
time labels, bracketed actions and all backup material.
Hyphenated words and contractions count as one word.

| Section | Speaker | Pages | Clock | Minutes | Words | Target at 105 wpm | Allowed range, +/-15% | Check |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Opening | LIU Xuyang | 1-3 | 00:00-03:00 | 3 | 271 | 315 | 268-362 | PASS |
| Background | WANG Zeqing | 4-7 | 03:00-10:00 | 7 | 654 | 735 | 625-845 | PASS |
| Methods | WU Aoqi | 8-14 | 10:00-18:00 | 8 | 748 | 840 | 714-966 | PASS |
| Results | ZHAO Jixin | 15-21 | 18:00-26:00 | 8 | 745 | 840 | 714-966 | PASS |
| Conclusion | LIU Xuyang | 22-24 | 26:00-30:00 | 4 | 387 | 420 | 357-483 | PASS |
| Total | Four speakers | 1-24 | 00:00-30:00 | 30 | 2,805 | 3,150 | All sections pass | PASS |

At 105 spoken words per minute, the text takes about 26 minutes 43 seconds.
The schedule leaves about 3 minutes 17 seconds for pointing, pauses and
handoffs. Across the full slots, the effective rates are 90.3, 93.4, 93.5,
93.1 and 96.8 words per minute respectively, within the requested tolerance.
These are word-count estimates, not measured rehearsal times.

## How to present

Use the Markdown script for rehearsal and the short frame notes as reminders.
Speak in short phrases, look up at sentence endings, and pause at each
bracketed pointing cue. On a formula slide, point to the term being discussed
and explain its job without reading every symbol. On a chart, name the axes
before describing the pattern. Read only the table values named in the script.
Rehearse with a visible clock and hand over at 03:00, 10:00, 18:00 and 26:00.
Use the remaining time in each slot for the marked pauses, not extra technical
detail. Finish on page 24 at 30:00 and keep it visible for questions.
Open pages 25-26 only when needed.

## Build

The requested command completed successfully:

```sh
cd slides/beamer/Group10
latexmk -xelatex -interaction=nonstopmode main.tex
cp main.pdf ../../Group10_Oral_PolyU_Beamer.pdf
```

The 24 talk titles and their order match the script. All 24 talk frames have
2-5 note sentences, and `main.tex` contains no Chinese characters.
No algorithm, theme file, result table or figure was changed for this delivery.
