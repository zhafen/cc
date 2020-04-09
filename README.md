# cc
An exploratory project to map out concept space using papers as data.

Publication class features include...
* Import of data from BibTex files using [bibtexparser](https://github.com/sciunto-org/python-bibtexparser).
* Import of ADS data using [the ads Python package](https://ads.readthedocs.io/en/latest/#the-ads-python-package). This includes...
- All references and citations.
- The latest citation information (no need to manually check if an arXiv paper has been published and update the citation).
* Parsing of BibTex annotations, including extraction of key concepts.

Atlas class features include...
* Easy bulk access to a full .bib of papers, including easy access to features included in the Publication class.
* A compilation of unique key concepts across a .bib, using natural language processing (via [nltk](https://www.nltk.org/)) to reduce the list of key concepts to unique key concepts, accounting for typos, missing spaces, capitalization, and plurals.
* Search papers for key concepts.
* Automatic approximate generation of relevant text for a key concept - in progress

