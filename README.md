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

Requires a TeX distribution (TeX Live / MacTeX) with `latexmk`.

```bash
cd book
make          # builds main.pdf
make watch    # continuous rebuild
make clean    # remove build artifacts
```

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
