# Submission Checklist

- Manuscript: `paper/jcs_main.tex`
- Bibliography: `paper/references.bib`
- Final PDF: `paper/jcs_main.pdf`

## Metadata

- Author shown as `Ruiqi Wang`
- Affiliation shown as `Department of Physics, University of California San Diego, La Jolla, CA, United States`
- Corresponding author footnote present
- Email shown as `riw010@ucsd.edu`
- No placeholder author/email text remains in the compiled PDF

## Citations and References

- No unresolved citation markers (`[?]` or `[? ?]`) remain
- References compile cleanly with readable spacing and punctuation
- Accented names render correctly: `Schölkopf`, `Schütt`, `Müller`, `Veličković`
- `TD-MPC2` capitalization is preserved in the compiled references
- BibTeX `empty pages` warnings for `hansen2024tdmpc2` and `li2018dcrnn` are resolved

## Figures and Tables

- All six figures render inline in the compiled PDF
- Figure 3 and Figure 4 footprint checked; both remain at `0.9\linewidth`
- Table 1 uses shortened display labels and remains readable
- Table 3 remains without the `source files` column in the main text
- Tables 4 and 5 remain readable in the compiled PDF
- Appendix E command block remains copyable and preserves literal double hyphens

## Availability

- `Data and Code Availability` includes the GitHub project URL: `https://github.com/2GyounnnG/causalworld`
- The section states that the manuscript introduces no new experiments in this version
- A versioned archival snapshot is noted as pending publication or repository release

## Appendix D

- Appendix D is renamed to `Claim traceability and supporting evidence`
- The appendix intro now frames the table as claim traceability rather than an internal evidence map

## Section Order

- `Conclusion` appears before `Data and Code Availability`
- `Data and Code Availability` remains before the appendix
- Appendix material remains after the main text

## Build Verification

- Recompiled with:
  - `pdflatex jcs_main.tex`
  - `bibtex jcs_main`
  - `pdflatex jcs_main.tex`
  - `pdflatex jcs_main.tex`
- Visual spot checks performed on the title page, figure pages, and table pages

## Remaining Known Issues

- No major LaTeX overfull or underfull warnings remain in the final `pdflatex` passes