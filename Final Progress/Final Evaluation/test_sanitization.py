"""Run the Final Evaluation text sanitization (V3) on sample messages and show every step.

The cleaning code is read from the notebook itself (cell A.3), and its patterns from M07's final_config.json
(the local copy in ../M07/m07_artifacts), so this tests exactly what runs on Colab.
Usage:  python3 test_sanitization.py                (uses the sample messages below)
        python3 test_sanitization.py messages.txt   (one message per line)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, 'Group_1_Email_Spam_Detection_Final_Evaluation.ipynb')
CONFIG = os.path.join(HERE, '..', 'M07', 'm07_artifacts', 'final_config.json')

SAMPLES = [
    "carolynn everett redresssk5paloaltocommonscom world may one person one person may world httpuaqdmotivationballcom thu 07 aug 2008 135316 0300",
    "help center info660508367168708400334271260866065842565402757531620311248823730000285682383counsellingonlineorgau 5222022 114033 pm sun 22 may 2022 234034 0000",
    "wood tom tomwoodkarchernacom dear colleague department performing system maintenance reconfiguration core switches performing major infrastructure upgrade serve",
    "nadine greenberg affablyuf10carrollextcom fastshippinginternetpricesfdaapproved httpnhk9ogbaylivefilestorecomy1phqwcigd0nqvzoaujij1d_1dxwrmo7jvedalxtb8g5m7hkb4i",
    "dwayne zimmerman capitalizekyn1painstandardscom pharmacyphentrimineuslicensed httpwmz84gbaylivefilestorecomy1p_guerurfqhsxx8z3i9gdda6rnynvryza9iuy7zqxx_v_wqv253",
    "guide thru answers questions laser vision correction good chance could throw glasses contacts away information tuaujryt",
    "13 1891 collect call alyssa milano 1824 small world songs",
    "amazing sxsxth hope use hostettler kucan liquefaranno nl ha maritavate got vi co din named parac odin x nax popular products http w net mypillsplace com x 8 tl",
    "utf8qapple20support ticket1541545idapplesupportcompany review case reference a1900xd180 id 1900x10d solutionid 12a9fjak94jf9ak recent summary existing case case",
    "maxine ramos seismologistslyg22wahcoitorg healthylifecertifiedhealth httpkk1vbwblulivefilestorecomy1p9yeouluxvtx6hddhacryfft5u4merx7oqdwnid6p1ugh0dwhpwijtyy5xzx",
]


def load_cleaning_code():
    cells = json.load(open(NOTEBOOK))['cells']
    source = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code' and 'def normalize_v3' in ''.join(c['source']))
    definitions = source[:source.index('start = time.perf_counter()')]
    namespace = {'re': re, 'np': np, 'pd': pd, 'config': json.load(open(CONFIG))}
    exec(definitions, namespace)
    return namespace


def main():
    ns = load_cleaning_code()
    texts = SAMPLES
    if len(sys.argv) > 1:
        texts = [line.rstrip('\n') for line in open(sys.argv[1], encoding='utf-8') if line.strip()]

    steps = [
        ('1 punctuation -> space', ns['PUNCTUATION_PATTERN'], ' '),
        ('2 header timestamp -> deleted', ns['TIMESTAMP_PATTERN'], ' '),
        ('3 link -> urltoken', ns['LINK_PATTERN'], ' urltoken '),
        ('4 www/com/net/org/edu/gov -> deleted', ns['WEB_PARTS_PATTERN'], ' '),
        ('5 number / escape token -> numtoken', ns['NUMBER_PATTERN'], ' numtoken '),
        ('6 month / weekday -> datetoken', ns['DATE_PATTERN'], ' datetoken '),
        ('7 stopwords -> deleted', ns['STOPWORD_PATTERN'], ' '),
    ]
    for number, text in enumerate(texts):
        print(f'\n=== Message {number} ===')
        print(f'{"original":38s} {text}')
        current = text.lower()
        for name, pattern, replacement in steps:
            new = re.sub(pattern, replacement, current)
            if ' '.join(new.split()) != ' '.join(current.split()):
                removed = [m.group(0) for m in re.finditer(pattern, current)]
                shown = ', '.join(repr(r.strip()) for r in removed[:6]) + (' …' if len(removed) > 6 else '')
                print(f'{name:38s} matched: {shown}')
            current = new
        final = ' '.join(current.split())
        same = final == ' '.join(ns['normalize_v3'](pd.Series([text])).iloc[0].split())
        print(f'{"V3 (what the model reads)":38s} {final}')
        print(f'{"":38s} {len(final.split())} words left; identical to notebook normalize_v3: {same}')


if __name__ == '__main__':
    main()
