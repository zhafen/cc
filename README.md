# cc
Data-centric exploration of literature.
This package is in development.

Documentation at [zhafen.github.io/cc/](https://zhafen.github.io/cc/)

## Features

### Data-Access Features
* Import of metadata from BibTex files using [bibtexparser](https://github.com/sciunto-org/python-bibtexparser).
* Import and storage of ADS metadata using [the ads Python package](https://ads.readthedocs.io/en/latest/#the-ads-python-package). This includes...
  * All references and citations and their metadata
  * The latest citation information (no need to manually check if an arXiv publication has been published and update the citation).
* Import and storage of arXiv source files (in-progress).

### Data-Processing Features
* Natural language processing of abstracts to extract key words according to word tagging.
* Identify unique words in a text, accounting for stemming and mispellings (but not for words that are too short, and therefore can turn into too many other words).
* Parsing of custom annotations.
* Multiple consistent methods for calculating the "angle" between two publications or between a publication and a collection of publications.
* Projection of abstracts into a virtual linear space
* Parsing and organizing of source tex files to allow for easier language analysis, including...
  - Breaking into sections, including the appendix
  - Easy sentence and work tokenization using NLTK.
  - Easy word tagging using NLTK.
  - Removing comments.
  - Handling \include statements
  - Handling macros
  - Changing ~ into whitespace unless escaped
  - Informative visual display of roughly-chunked sentences.

### Practicing Scientist Utility Features:
* Automatic approximate generation of relevant text for a given concept.
* Generate a bibtex file from ADS calls.

## Why does this package exist?

Modern science has expanded to be incomprehensible for a single person.
A quick calculation to demonstrate this:
As a scientist you may be able to read and comprehend up to roughly 10 papers per day and still get other work done.
However, on the astrophysics arxiv alone dozens to more than one hundred publications come out per day.
Specializing in an area makes it feasible to read all relevant papers, but does not typically allow the scientist to see the larger view.
This package is aimed at addressing these issues using the wealth of paper data and metadata easily available online.
