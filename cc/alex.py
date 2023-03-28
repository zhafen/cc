'''Function from Alex to move to the appropriate location at a future date.
'''

import os

fname = os.path.join(os.environ['HOME'],'Desktop','jose_bib.bib')
new_fname = os.path.join(os.environ['HOME'],'Desktop','jose_bib_adj.bib')

def get_first_author(entry):
    author = entry.split('\n')[1]
    author = author.split(',')[0].split('=')[1]

    author = author.replace('{','').replace('}','').replace(' ','')
    if '\\' in author:
       index = author.index('\\')
       author = author[:index]+author[index+2:]
    return author

def get_year(entry):
    index = entry.index('{')+1
    year = entry[index:index+4]
    return year.replace(' ','')

def replace_cite_key(entry):
    '''Replaces ADS citation key with a more intelligible one.
    API_extension::no_change
    '''
    key = "%s%s"%(get_first_author(entry),get_year(entry))

    index = entry.index('{')+1
    findex = entry.index(',')
    entry = entry[:index] + key + entry[findex:]
    return entry

with open(fname,'r') as handle:
    big_guy = "".join(handle.readlines())

    entries = big_guy.split("@")[1:]

    for i,entry in enumerate(entries):
       entries[i] = replace_cite_key(entry)

    big_guy = "@".join(entries)
    with open(new_fname,'w') as handle:
       handle.write("@"+big_guy)
