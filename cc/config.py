
nltk = {
    # 1st tier is basically adjectives, nouns, and verbs
    # 2nd tier is everything else
    'tag_tier': {
        1: [
            'FW',
            'JJ',
            'JJR',
            'JJS',
            'NN',
            'NNP',
            'NNPS',
            'NNS',
            'RB',
            'RBR',
            'RBS',
            'VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
            'VBZ',
        ],
        2: [
            '$',
            "''",
            '(',
            ')',
            ',',
            '--',
            '.',
            ':',
            'CC',
            'CD',
            'DT',
            'EX',
            'IN',
            'LS',
            'MD',
            'PDT',
            'POS',
            'PRP',
            'PRP$',
            'RP',
            'SYM',
            'TO',
            'UH',
            'WDT',
            'WP',
            'WP$',
            'WRB',
            '``',
        ],
    },
}
'''Natural language processing toolkit settings.'''
