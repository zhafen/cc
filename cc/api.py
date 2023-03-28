'''This file contains code for interfacing cc with various APIs.

Required changes below. Search the files for the given flag name to find relevant areas that need to be changed.
The below text is not guaranteed to contain everything we need to change,
and it wouldn't be surprising if I misclassified or classified twice.

get_data_via_api:
    The most basic functionality we use.
    There will be three main types of calls I expect to show up.
    get_data_via_api_general --- the base call that is what shows up most places in the code.
    get_ads_data --- the ADS specific call. Probably should only be used by get_data_via_api_general
    get_ss_data --- The SS specific call. Same deal as ADS.

process_data:
    Take downloaded data and do something with it. Typically relies on the specific format of the downloaded data.

to_and_from_bibcodes:
    Bibcodes are the identifiers ADS uses. cc has the functionality to turn a list of bibcodes into an Atlas,
    and related functionality for turning bibcodes into a bibtex file.
    The general analog to this will probably need to be from_ids and similar.

random_publications:
    The functionality for retrieving random publications is currently ADS specific.

publication_date:
    The publication date is currently set to the date the publication was posted on arXiv.
    We'll need to make it more general.

default_bib_name:
    Change the default name of the .bib file created during algorithmic retrieval of data.
    Current name is cc_ads.bib.

maybe_unnecessary:
    This code should probably be removed after the extension is finished.

no_change:
    This code uses ADS, but should not be touched. Probably because it's not used.
'''