# cc
Helping scientists explore the literature.

Why does this package exist?
Modern science has expanded to the point where it is incomprehensible by a single person.
A quick calculation to demonstrate this:
Let's say as a scientist you can take the time to read and comprehend 5 papers per day (being generous).
However on astro-ph alone ~40-80 papers come out per day.
Specializing in an area makes it feasible to read all relevant papers, but does not typically allow the scientist to see the larger view.
This package is aimed at addressing these issues using the wealth of paper data and metadata easily available online through ADS and arXiv.

Utility Features:
* Import of data from BibTex files using [bibtexparser](https://github.com/sciunto-org/python-bibtexparser).
* Import and storage of ADS data using [the ads Python package](https://ads.readthedocs.io/en/latest/#the-ads-python-package). This includes...
- All references and citations.
- The latest citation information (no need to manually check if an arXiv paper has been published and update the citation).
* A function for identifying unique words, accounting for stemming and mispellings (but not for words that are too short, and therefore can turn into too many other words)
* Generate a bibtex file using unique calls to ADS as input.

Exploration Features:
* Natural language processing of abstracts to extract important concepts.
* Parsing of annotations to extract important concepts.
* Multiple consistent methods for calculating the "angle" between two publications or between a publication and a collection of publications.
* Easy import of papers that cite or are referenced by a given paper.

Paper-Writing Features:
* Automatic approximate generation of relevant text for a key concept.

########################################################################

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
