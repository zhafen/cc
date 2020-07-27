# cc
An exploratory project to map out concept space using papers as data.

Utility features include...
* A function for identifying unique words, accounting for stemming and mispellings (but not for words that are too short, and therefore can turn into too many other words)

Publication class features include...
* Import of data from BibTex files using [bibtexparser](https://github.com/sciunto-org/python-bibtexparser).
* Import of ADS data using [the ads Python package](https://ads.readthedocs.io/en/latest/#the-ads-python-package). This includes...
- All references and citations.
- The latest citation information (no need to manually check if an arXiv paper has been published and update the citation).
* Parsing of BibTex annotations, including extraction of key concepts.
* Parsing of abstracts using natural language processing.

Atlas class features include...
* Easy bulk access to a full .bib of papers, including easy access to features included in the Publication class.
* Easy saving and storing of ADS data for the full .bib of papers.
* A compilation of unique key concepts across a .bib, using natural language processing (via [nltk](https://www.nltk.org/)) to reduce the list of key concepts to unique key concepts, accounting for typos, missing spaces, capitalization, and plurals.
* Find papers that discuss key concepts and extract relevant text.
* Automatic approximate generation of relevant text for a key concept.

As Sarah put it:
You could put in, say, halo virialization, and it will return all the papers you've ever read that involved halo virialization, as well as what you learned from them.

Inner product:
* Multiple methods for evaluating the inner product between papers.
* Abstract similarity method uses natural language processing to compare similarity between abstracts. If the abstracts aren't included in the output bibliography then they are automatically downloaded from ADS.

Visualizations:
* cospsi plot: visualize the "angle" between two papers, between a paper and a collection of papers, or between a paper and vector of words.
