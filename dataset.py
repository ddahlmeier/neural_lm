"""
Dictionary that maps words to integer ids
"""

import util


def load_corpus(file_name, vocab=None):
    """
    load raw text corpus, sentence per line into list of lists
    """
    corpus = []
    with open(file_name, 'r') as fin:
        for line in fin:
            if len(line.split()) > 0:
                if (vocab == None):
                    corpus.append(line.split())
                else:
                    corpus.append([w for w in line.split() if w in vocab])
    return corpus


class Dictionary:
    """
    Mutable associative container that maps word to integer ids and vice versa.
    """
    def __init__(self, vocab=[], unk=None):
        # index 0 is reserved for the unknown word, normal words start at index 1
        self.vocab = [unk] + vocab
        self.map_words_to_ids = dict(map(lambda x : (x[1], x[0]+1), enumerate(vocab)))

    @classmethod
    def from_file(cls, filename, unk=None):
        with open(filename, 'r') as fin:
            vocab = [ line.strip() for line in fin.readlines() if line.strip() != "" and len(line.split()) == 1]
        return cls(vocab, unk)

    @classmethod
    def from_corpus(cls, corpus, unk=None):
        vocab = util.uniq(reduce(lambda x, y : x + y, corpus))
        return cls(vocab, unk)

    def corpus_words_to_ids(self, corpus, update_dict = False):
        corpus_ids = [self.doc_words_to_ids(doc, update_dict=update_dict) for doc in corpus]
        return corpus_ids

    def doc_words_to_ids(self, doc, update_dict = False):
        doc_ids = [self.lookup_id(word, update_dict = update_dict) for word in doc if self.lookup_id(word, update_dict) != None]
        return doc_ids


    def lookup_id(self, word, update_dict = False):
        try:
            word_id = self.map_words_to_ids[word]
            return word_id
        except KeyError:
            if update_dict:
                return self.add_word(word)
            else:
                # index 0 is reserved for unknown word
                return 0

    def add_word(self, word):
        word_id = len(self.vocab)
        self.vocab.append(word)
        self.map_words_to_ids[word] = word_id
        return word_id

    def lookup_word(self, word_id):
        try:
            word = self.vocab[word_id]
            return word
        except IndexError:
            return None

    def size(self):
        return len(self.vocab)
