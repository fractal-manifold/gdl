#!/usr/bin/env bash
#
# Build all book PDFs (+ KDP wraparound covers) and publish them as a
# date-stamped GitHub release.
#
#   scripts/release.sh            # tag is today's date: book-YYYY-MM-DD
#   scripts/release.sh 2026-06-27 # override the date explicitly
#
# Requirements: latexmk + a TeX distribution, ImageMagick (magick),
# poppler-utils (pdfinfo), and the GitHub CLI (gh) authenticated against
# the repo. portada.jpg must be present at the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

DATE="${1:-$(date +%Y-%m-%d)}"
TAG="book-$DATE"
TITLE="Geometric Deep Learning — $DATE"

echo "==> Building PDFs into out/ ..."
make -C book dist

read -r -d '' NOTES <<EOF || true
Book PDFs built on $DATE.

| File | Edition |
|------|---------|
| \`geometric-deep-learning.pdf\` | A4 reading edition (English) |
| \`geometric-deep-learning-print.pdf\` | 7×10 in print edition (English) |
| \`geometric-deep-learning-es.pdf\` | A4 reading edition (Spanish) |
| \`geometric-deep-learning-print-es.pdf\` | 7×10 in print edition (Spanish) |
| \`geometric-deep-learning-cover.pdf\` | KDP wraparound cover (English) |
| \`geometric-deep-learning-cover-es.pdf\` | KDP wraparound cover (Spanish) |
EOF

echo "==> Publishing release $TAG ..."
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "    Release $TAG already exists — refreshing assets and notes."
  gh release upload "$TAG" out/*.pdf --clobber
  gh release edit "$TAG" --title "$TITLE" --notes "$NOTES"
else
  gh release create "$TAG" out/*.pdf --title "$TITLE" --notes "$NOTES" --latest
fi

echo "==> Done. https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
