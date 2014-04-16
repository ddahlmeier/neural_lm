#!/usr/bin/env python

"""
Pre-process the PTB WSJ data for language modelling
 - Extract word tokens from POS tagged data
 - lowercase tokens
 - reduce vocabulary to 10k word types

usage: wsj_preprocess.py wsj_dir wsj_path, train_file, dev_file, test_file
"""


import gzip
from collections import Counter

vocab_sz = 10**4

def smart_open(file_name, mode='rb'):
    if file_name.endswith('.gz'):
        f = gzip.open(file_name, mode)
    else:
        f = open(file_name, mode)
    return f
        

def sentences(source):
    buffer = []
    for line in source:
        line = line.strip()
        if line == '':
            continue
        elif line == "======================================":
            if buffer:
                yield buffer
            buffer = []
        else:
            buffer = buffer + [word for word in line.split() if '/' in word]
    if buffer:
        yield buffer            
    
def get_token(word_pos, delim = '/'):
    return word_pos[:word_pos.rindex(delim)]
    
def tokens(sentence):
    return [get_token(word_pos) for word_pos in sentence]

def lower_case(sentence):
    return [word.lower() for word in sentence]

def replace_unknown(sentence, vocab, unk = '-unk-'):
    return [word if word in vocab else unk for word in sentence]


if __name__ == "__main__":
    import sys
    from os import listdir
    from os.path import isfile, join
    
    if len(sys.argv) != 5:
        print "usage: python wsj_process.py wsj_path, train_file, dev_file, test_file"
        sys.exit(1)
    wsj_path, train_file, dev_file, test_file = sys.argv[1:]
    train_set = []
    dev_set = []
    test_set = []
    print "Load data.."
    for wsj_file in (f for f in listdir(wsj_path) if isfile(join(wsj_path, f)) and f.startswith('wsj_')):
        wsj_sec = int(wsj_file[4:6])
        fin = smart_open(join(wsj_path, wsj_file))
        lines = [ lower_case(tokens(sentence)) for sentence in sentences(fin) ]
        if 0 <= wsj_sec <= 20:
            train_set = train_set + lines
        elif 21 <= wsj_sec <= 22:
            dev_set = dev_set + lines
        elif 23 <= wsj_sec <= 24:
            test_set = test_set + lines
        else:
            print "Unknown section :", wsj_sec
            sys.exit(1)
        fin.close()        
    # create vocabulary from train set
    print "Create vocabulary.."
    word_counts = Counter((word for sentence in train_set for word in sentence))
    vocab = {word for word, count in word_counts.most_common(vocab_sz)}
    print "Write output.."
    for data, file_out in zip([train_set, dev_set, test_set], [train_file, dev_file, test_file]):
        fout = smart_open(file_out, 'w')
        for sentence in (replace_unknown(line, vocab) for line in data):
            fout.write(' '.join(sentence) + '\n')
        fout.close()
