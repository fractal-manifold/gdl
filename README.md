# Geometric Deep Learning

Source for the book **Geometric Deep Learning** and its companion website
published at **[gdl.fractalmanifold.com](https://gdl.fractalmanifold.com)**.

## Repository layout

```
.
├── book/              # LaTeX sources for the book (PDF)
│   ├── main.tex       # master document
│   ├── preamble.tex   # packages, macros, theorem environments
│   ├── frontmatter/   # title page, abstract, preface
│   ├── chapters/      # one file per chapter
│   ├── figures/       # images / TikZ assets
│   ├── bib/           # BibTeX references
│   └── Makefile       # build helpers (latexmk)
│
├── web/               # Astro website (static, deployed to GitHub Pages)
│   ├── src/
│   │   ├── content/   # chapter content (Markdown)
│   │   ├── layouts/
│   │   ├── components/
│   │   └── pages/
│   └── public/CNAME   # custom domain for GitHub Pages
│
└── .github/workflows/ # CI: build & deploy the site
```

## Building the book (PDF)

Requires a TeX distribution (TeX Live / MacTeX) with `latexmk`, plus
`ImageMagick` and `poppler-utils` (`pdfinfo`) for the covers.

The book ships in two editions × two languages, plus a print cover:

```bash
cd book
make            # all four interior editions (EN/ES × A4/print)
make screen     # A4 reading edition, English (also: print, screen-es, print-es)
make cover      # Amazon KDP wraparound cover, English (also: cover-es)
make watch      # continuous rebuild of the screen edition
make clean      # remove build artifacts
```

Every edition carries an image **front cover** (page 1, from `portada.jpg`).
The `cover` / `cover-es` targets produce the separate **full wraparound**
print cover (`[back | spine | front]`), sizing the spine from the matching
print edition's page count (KDP white-paper formula).

### Collecting and releasing PDFs

```bash
make dist                       # build everything into the repo-level out/
scripts/release.sh              # build + publish a date-stamped GitHub release
scripts/release.sh 2026-06-27   # ... with an explicit date
```

`make dist` renames the PDFs and gathers them in `out/`.
`scripts/release.sh` runs `make dist` and publishes them to a release tagged
`book-YYYY-MM-DD` via the GitHub CLI (`gh`). Pushing changes under `book/` (or
`portada.jpg`) to `main` also triggers `.github/workflows/book.yml`, which
builds the PDFs and publishes the same date-stamped release automatically.

## Running the website locally

Requires Node.js 20+.

```bash
cd web
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in web/dist
```

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the
Astro site and publishes it to GitHub Pages. The custom domain is configured
through `web/public/CNAME`.

## License

See [LICENSE](LICENSE).
