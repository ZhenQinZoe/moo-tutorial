# Multi-Objective Optimization for Modern Machine Learning: Foundations, Recent Advances, and Emerging Applications

A self-contained static tutorial website prepared from the supplied HackMD workspace. The site can be previewed locally or published with the included GitHub Pages workflow.

## Preview

Open `dist/index.html` directly in a browser, or run:

```sh
npm ci
npm run build
npm run preview
```

Then open `http://localhost:4173`.

## Content preservation

`content/` contains the 16 original Markdown notes, copied without rewriting. The manifest records each source URL and SHA-256 checksum. Every build verifies these checksums before rendering. Original wording, equations, author annotations, and existing placeholders are retained. Each source `[TOC]` displays all subsection headings in their original order, without repeating the page title. At the user’s request, “(Kevin)” is removed from displayed headings; the source files remain unchanged.

At the user's request, local Reference/References headings and blocks are removed from display, with all works collected on one deduplicated **References** page. `references/verified.json` records 185 distinct works, their verified metadata, original publication links, evidence, and version differences. Nine papers cited in the body but absent from the original bibliography have been added to this shared list. The MGDA-FA placeholder is mapped to its actual paper, already cited elsewhere. At the user’s request, sentence-initial narrative citations retain author wording (for example, Ye et al. [165]); other author/year labels become numeric citations, and parentheses around citation-only groups are removed. Method names and pinpoint references are retained. Clickable in-body citations use globally consistent English square-bracket numbers such as [1], and the shared list is numbered 1–185. Existing local numbers are remapped to the same shared numbering without duplicate adjacent markers; reference titles link to original source pages. CLEAR's unidentified “OUR PREPRINT” has a red **TODO**, as requested, with no invented authors or URL.

`references/AUDIT.md` documents the bibliographic check and its limitations. This is an existence/metadata check, not a scientific claim-by-claim or proof audit. `references/site-validation.json` records the source checksums and interface/link checks. Earlier browser checks are retained as historical validation; the latest citation-display check is `references/narrative-citation-validation.json`; the section/navigation check is `references/numbered-ui-validation.json`.

The primary navigation uses Chapter 0–5, followed by the standalone, unnumbered **References** section immediately after Chapter 5: Applications. The site name is **Multi-Objective Optimization for Modern Machine Learning: Foundations, Recent Advances, and Emerging Applications**. Outline, Foundation notes, View source, and HackMD footer controls are removed from the interface. Auxiliary source pages remain available for existing internal links but are not promoted in navigation or search. References use a compact numbered list with one original-paper link per entry. The default HackMD team welcome note is service boilerplate and is not part of the tutorial.

## Sharing

The author portraits in `assets/authors/` are original downloads from the authors' official homepages, except Zhiyao Zhang's replacement photograph supplied by the user. `authors.json` records each source and display crop. Lisha Chen uses the formal portrait selected by the user. CSS frames use direct square crops centered on each author's head within the original image bounds, preserving the image's aspect ratio without padding or blurred backgrounds. The student-first surname order, equal-contribution marks, and Jia Liu's final position are preserved.

The `dist/` directory is the complete website. It includes its math, diagram, and search libraries and can be shared as a ZIP or uploaded to any static host, including GitHub Pages. Extract the ZIP before opening `index.html`. A localhost address cannot be accessed by other people.

## Source asset gaps

All main-chapter illustrations are packaged locally. Three relative PNG references were matched to the same original uploaded illustrations; those aliases and their provenance are recorded in `assets/manifest.json`.

One additional illustration, `HJX9zpkCfe.png`, appears in two foundation notes and remains at its original HackMD URL because its download was unavailable. It requires network access and may not load outside the authenticated source session. Its exact source reference is retained; no substitute drawing was created.

Three auxiliary download links were already missing from the supplied notes: `assets/classical-constrained-moo.svg`, `assets/generate-classical-constrained-moo.mjs`, and `assets/cagrad-manipulation.svg`. Their labels and original targets are retained. These gaps do not affect chapter navigation, references, or the main-chapter figures.

## Build

`build.mjs` renders the source files with Markdown-it, MathJax, and Mermaid. `theme.css` controls layout, and `app.js` implements navigation, search, reading progress, and figure enlargement. The design follows the supplied reference's academic documentation layout.

Third-party libraries retain their packaged licensing information in `node_modules`; redistributed license notices are provided with the built website.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` verifies the source checksums, builds the site, and publishes it from `dist/`. In the repository settings, set **Pages → Source → GitHub Actions**. The website is independent of the account’s personal homepage.

## Reading and rendering

Chapter titles consistently use Chapter 0–5. Complete inline contents are expanded by default and can be collapsed with a keyboard-accessible native control. Bare LaTeX equation environments are preserved before Markdown parsing. Long formulas and tables scroll inside their own containers on small screens; overflowing formulas are keyboard-focusable.

`references/reading-validation.json` and `references/reading-browser-validation.json` record the latest checks: all 2,842 math expressions compiled without errors, all original source checksums matched, complete contents and citation targets were preserved, and the tested mobile/desktop pages had no horizontal page overflow.
