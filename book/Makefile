# Build the book with latexmk.
# The source is bilingual (English / Spanish) and ships in two editions,
# both selected via small wrapper files:
#   screen.tex     -> A4 reading edition, English (default)
#   print.tex      -> 7x10 in print-on-demand edition, English
#   screen-es.tex  -> A4 reading edition, Spanish
#   print-es.tex   -> 7x10 in print-on-demand edition, Spanish
LATEXMK = latexmk -pdf -bibtex -interaction=nonstopmode

.PHONY: all en es screen print screen-es print-es watch watch-es clean

all: en es
en: screen print
es: screen-es print-es

screen: screen.pdf
print: print.pdf
screen-es: screen-es.pdf
print-es: print-es.pdf

DEPS = main.tex preamble.tex lang/en.tex lang/es.tex

screen.pdf: screen.tex $(DEPS)
	$(LATEXMK) screen.tex

print.pdf: print.tex $(DEPS)
	$(LATEXMK) print.tex

screen-es.pdf: screen-es.tex $(DEPS)
	$(LATEXMK) screen-es.tex

print-es.pdf: print-es.tex $(DEPS)
	$(LATEXMK) print-es.tex

watch:
	$(LATEXMK) -pvc screen.tex

watch-es:
	$(LATEXMK) -pvc screen-es.tex

clean:
	latexmk -C screen.tex print.tex screen-es.tex print-es.tex
	rm -f *.bbl *.run.xml
