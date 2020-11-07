.. cc documentation master file, created by
   sphinx-quickstart on Sat Nov  7 10:03:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cc
==============================

Exploring the topography of scientific literature.

Installation
============

As cc is still in active development and primarily for contributor use it's recommended to install via cloning and then running pip.
In the location where you want to install do:

.. code-block:: console

    git clone git@github.com:zhafen/cc.git
    cd cc
    pip install -e .

Next, download data used by the Natural Language Processing Toolkit.

.. code-block:: console

    python -m nltk.downloader all

Usage
=====

For an interactive tutorial and example, open `the charting.ipynb notebook in the examples folder of the repository <https://github.com/zhafen/cc/blob/master/examples/charting.ipynb>`__.

.. toctree::
   :maxdepth: 1

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
