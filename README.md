# MOO Tutorial

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

`content/` contains the 16 original Markdown notes, copied without rewriting. The manifest records each source URL and SHA-256 checksum. Every build verifies these checksums before rendering. Original wording, equations, author annotations, references, and existing placeholders are retained. Navigation, styles, math typesetting, and local copies of image assets are presentation changes only.

Six main chapters are in the primary navigation. The roadmap is under **Outline**, and the nine separately stored foundation notes remain accessible under **Foundation notes**. The default HackMD team welcome note is service boilerplate and is not part of the tutorial.

## Sharing

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
