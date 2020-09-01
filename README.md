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

Other features:
* Parsing and organizing of source tex files, includin...
- Handling \include statements
- Breaking into sections, including the appendix
- Easy sentence and work tokenization using NLTK.
- Easy word tagging using NLTK.
- Separating comments
