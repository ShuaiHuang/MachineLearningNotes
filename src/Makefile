.PHONY : all pdf clean release

project-name = Machine-Learning-Reading-Note
file-name = $(project-name).pdf

all: pdf clean

pdf:
	xelatex $(project-name).tex && \
	xelatex $(project-name).tex

clean:
	rm -rf *.toc *.aux *.log *.out *.blg *.bbl *.dvi *.ps *.gz

release:
ifeq ($(file-name), $(wildcard $(file-name)))
	mv $(file-name) ../$(file-name)
else
	$(warning "pdf file not exist!")
endif