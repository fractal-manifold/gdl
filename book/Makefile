# Build the book with latexmk.
# The source is bilingual (English / Spanish) and ships in two editions,
# both selected via small wrapper files:
#   screen.tex     -> A4 reading edition, English (default)
#   print.tex      -> 7x10 in print-on-demand edition, English
#   screen-es.tex  -> A4 reading edition, Spanish
#   print-es.tex   -> 7x10 in print-on-demand edition, Spanish
#
# Every interior gets an image front cover (page 1). The separate Amazon
# KDP wraparound cover is built with `make cover` / `make cover-es`; its
# spine width is computed from the matching print interior's page count.
# The book builds with lualatex so the cover can use the Bruno Ace font.
LATEXMK = latexmk -lualatex -bibtex -interaction=nonstopmode
# The standalone KDP wrap cover (also fontspec/Bruno Ace, forced rebuild).
COVERMK = latexmk -g -lualatex -interaction=nonstopmode

# --- cover artwork (all derived from ../portada.jpg) ----------------
# ImageMagick 7 ships `magick`; IM6 (e.g. Debian CI) only ships `convert`.
MAGICK        := $(shell command -v magick 2>/dev/null || command -v convert 2>/dev/null)
SRC_ART        = ../portada.jpg
KDP_PPP        = 0.002252   # KDP white paper: spine inches per page
SPINE_TEXT_MIN = 100        # min interior pages before printing spine text
BLEED          = 0.125      # KDP bleed (in) on the outer edges

.PHONY: all en es screen print screen-es print-es watch watch-es clean cover cover-es dist figures

all: en es
en: screen print
es: screen-es print-es

screen: screen.pdf
print: print.pdf
screen-es: screen-es.pdf
print-es: print-es.pdf

# Every source file the editions depend on, so `make` rebuilds whenever any
# content changes (latexmk also tracks inputs, but make must invoke it first).
DEPS = main.tex preamble.tex frontmatter/cover.tex \
       $(wildcard lang/*.tex) \
       $(wildcard frontmatter/*/*.tex) \
       $(wildcard chapters/*/*.tex) \
       bib/references.bib figures/.stamp

# --- Figures ---------------------------------------------------------
# Source-only policy: figures/ versions just the TikZ standalone sources
# (*.tex), the matplotlib/TikZ generators (generate_*.py) and a few raster
# assets. The compiled *.pdf, the script-produced *.png and the two
# generator-emitted *.tex are build artifacts (git-ignored) regenerated
# here. Needs python3 + numpy/matplotlib/networkx and pdflatex (both in the
# CI TeX Live image once the Python packages are installed).
FIGMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
figures: figures/.stamp
figures/.stamp: $(wildcard figures/*.py) $(wildcard figures/*.tex)
	@echo ">> regenerating figures (python generators + standalone TikZ)"
	cd figures && for s in generate_*.py; do MPLBACKEND=Agg python3 "$$s"; done
	cd figures && for t in *.tex; do $(FIGMK) "$$t" >/dev/null; done
	@touch $@

# Digital front-cover rasters: mirrored (-flop) and cropped to the page
# aspect at 300 dpi (East gravity keeps the figure on the right).
figures/cover-front-print.jpg: $(SRC_ART)
	$(MAGICK) $(SRC_ART) -flop -resize "2100x3000^" -gravity East -extent 2100x3000 \
	  -units PixelsPerInch -density 300 $@
figures/cover-front-screen.jpg: $(SRC_ART)
	$(MAGICK) $(SRC_ART) -flop -resize "2481x3508^" -gravity East -extent 2481x3508 \
	  -units PixelsPerInch -density 300 $@

screen.pdf: screen.tex $(DEPS) figures/cover-front-screen.jpg
	$(LATEXMK) screen.tex

print.pdf: print.tex $(DEPS) figures/cover-front-print.jpg
	$(LATEXMK) print.tex

screen-es.pdf: screen-es.tex $(DEPS) figures/cover-front-screen.jpg
	$(LATEXMK) screen-es.tex

print-es.pdf: print-es.tex $(DEPS) figures/cover-front-print.jpg
	$(LATEXMK) print-es.tex

# --- Amazon KDP wraparound cover ------------------------------------
# One sheet [back | spine | front] sized to the interior. Generates
# cover-dimensions.tex and figures/cover-wrap.jpg, then builds cover.tex.
# $(1) = language (en|es)   $(2) = matching print interior PDF
define build_wrap
	@set -e; PAGES=$$(pdfinfo $(2) | awk '/^Pages:/{print $$2}'); \
	SPINE=$$(awk "BEGIN{printf \"%.4f\", $$PAGES*$(KDP_PPP)}"); \
	COVERW=$$(awk "BEGIN{printf \"%.4f\", 0.25+14+$$PAGES*$(KDP_PPP)}"); \
	WPX=$$(awk "BEGIN{printf \"%.0f\", (0.25+14+$$PAGES*$(KDP_PPP))*300}"); \
	HPX=$$(awk "BEGIN{printf \"%.0f\", 10.25*300}"); \
	if [ $$PAGES -ge $(SPINE_TEXT_MIN) ]; then SP='\spinetexttrue'; else SP='\spinetextfalse'; fi; \
	printf '\\def\\coverlang{%s}\n\\def\\bleed{%sin}\n\\def\\Bn{%s}\n\\def\\spinewidth{%sin}\n\\def\\Sn{%s}\n\\def\\coverW{%sin}\n\\def\\coverH{10.25in}\n%s\n' \
	  $(1) $(BLEED) $(BLEED) $$SPINE $$SPINE $$COVERW "$$SP" > cover-dimensions.tex; \
	$(MAGICK) $(SRC_ART) -flop -resize "$${WPX}x$${HPX}^" -gravity East -extent $${WPX}x$${HPX} \
	  -units PixelsPerInch -density 300 figures/cover-wrap.jpg; \
	$(COVERMK) cover.tex; \
	mv cover.pdf cover-$(1).pdf; \
	echo ">> cover-$(1).pdf : $$PAGES pp, spine $$SPINE in, sheet $$COVERW x 10.25 in"
endef

cover: print.pdf cover.tex lang/en.tex
	$(call build_wrap,en,print.pdf)

cover-es: print-es.pdf cover.tex lang/es.tex
	$(call build_wrap,es,print-es.pdf)

# --- Distribution ----------------------------------------------------
# Build every edition + both KDP covers and collect them, renamed for
# humans, into the repo-level out/ directory (the release payload).
# Interiors build in parallel (under -j); the two cover builds are
# serialized via sequential sub-makes because they share intermediate
# files (cover.pdf, cover-dimensions.tex, figures/cover-wrap.jpg).
OUT = ../out
dist: all
	$(MAKE) cover
	$(MAKE) cover-es
	mkdir -p $(OUT)
	cp screen.pdf    $(OUT)/geometric-deep-learning.pdf
	cp print.pdf     $(OUT)/geometric-deep-learning-print.pdf
	cp screen-es.pdf $(OUT)/geometric-deep-learning-es.pdf
	cp print-es.pdf  $(OUT)/geometric-deep-learning-print-es.pdf
	cp cover-en.pdf  $(OUT)/geometric-deep-learning-cover.pdf
	cp cover-es.pdf  $(OUT)/geometric-deep-learning-cover-es.pdf
	@echo "==> PDFs collected in $(OUT)/"
	@ls -1 $(OUT)

watch:
	$(LATEXMK) -pvc screen.tex

watch-es:
	$(LATEXMK) -pvc screen-es.tex

clean:
	latexmk -C screen.tex print.tex screen-es.tex print-es.tex cover.tex
	rm -f *.bbl *.run.xml cover-dimensions.tex cover-en.pdf cover-es.pdf \
	  figures/cover-front-print.jpg figures/cover-front-screen.jpg figures/cover-wrap.jpg
	# Generated figure artifacts (regenerated by `make figures`).
	-cd figures && latexmk -C *.tex >/dev/null 2>&1
	rm -f figures/.stamp figures/*.pdf \
	  figures/activation_functions.png figures/cnn_to_spectral_graph.png \
	  figures/edge_detection_kernels.png figures/eigenvalue_spectrum.png \
	  figures/gauge_transport_cylinder.png figures/gaussian_smoothing_comparison.png \
	  figures/graph_fourier_transform.png figures/hyperbolic_embedding.png \
	  figures/laplacian_eigenvectors.png figures/loss_landscapes.png \
	  figures/manifold_hypothesis.png figures/sharpening_effect.png figures/so2_irreps.png \
	  figures/graph_fourier_transform.tex figures/laplacian_eigenvectors.tex
