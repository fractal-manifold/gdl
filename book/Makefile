# Build the book with latexmk.
# Two editions share one source, selected via wrapper files:
#   screen.tex -> A4 reading edition (default)
#   print.tex  -> 7x10 in print-on-demand edition
LATEXMK = latexmk -pdf -bibtex -interaction=nonstopmode

.PHONY: all screen print watch clean

all: screen print

screen: screen.pdf
print: print.pdf

screen.pdf: screen.tex main.tex preamble.tex
	$(LATEXMK) screen.tex

print.pdf: print.tex main.tex preamble.tex
	$(LATEXMK) print.tex

watch:
	$(LATEXMK) -pvc screen.tex

clean:
	latexmk -C screen.tex print.tex
	rm -f *.bbl *.run.xml
