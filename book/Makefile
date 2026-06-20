# Build the book with latexmk
MAIN = main

.PHONY: all watch clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	latexmk -pdf -bibtex -interaction=nonstopmode $(MAIN).tex

watch:
	latexmk -pdf -bibtex -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml
