# File: Makefile
SHELL           = /bin/bash
PROJECT         = thesis
BUILD_DIR       = build

.PHONY          = all clean

.PRECIOUS: $(PROJECT).ps

all: pdf

pdf: $(PROJECT).pdf

%.pdf: %.tex Makefile *.tex
	mkdir -p $(BUILD_DIR)
	latexmk -shell-escape -bibtex -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(<) 
# 	cp $(BUILD_DIR)/$(PROJECT).pdf .

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.lof *.log *.out *.synctex.gz *.toc

force:
	$(MAKE) -W $(PROJECT).tex
