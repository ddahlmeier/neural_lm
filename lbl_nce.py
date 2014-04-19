"""

The log-bilinear language model with noise constrastive estimation 
from (Mnih and Teh, ICML 2012)

References:
 A fast and simple algorithm for training neural probabilistic language models. 
 Andriy Mnih and Yee Whye Teh.
 International Conference on Machine Learning 2012 (ICML 2012) 

Usage: lbl.py [--verbose] [--word_dim WORD_DIM] [--context_sz CONTEXT_SZ] 
              [--learn_rate LEARN_RATE] [--rate_update RATE_UPDATE] 
              [--epochs EPOCHS] [--batch_size BATCH_SZ] [--seed SEED]  
              [--patience PATIENCE] [--patience_incr PATIENCE_INCR] 
              [--improvement_thrs IMPR_THRS] [--validation_freq VALID_FREQ] 
              [--noise_samples NOISE_SAMPLES]
              <train_data> <dev_data> [<test_data>]

Arguments:
 train_data       training data of tokenized text, one sentence per line.
 dev_data         development data of tokenized text, one sentence per line.
 test_data        test data of tokenized text, one sentence per line.

Options:
    -v, --verbose                                    Print debug information
    -K WORD_DIM, --word_dim=WORD_DIM                 dimension of word embeddings [default: 100]
    -n CONTEXT_SZ, --context_sz=CONTEXT_SZ           size of n-gram context window [default: 2]
    -l LEARN_RATE, --learn_rate=LEARN_RATE           initial learning rate parameter [default: 1]
    -u RATE_UPDATE, --rate_update=RATE_UPDATE        learning rate update: 'simple', 'adaptive' [default: simple]
    -e EPOCHS, --epochs=EPOCHS                       number of training epochs [default: 10]
    -b BATCH_SIZE, --batch_size=BATCH_SIZE           size of mini-batch for training [default: 100]
    -s SEED, --seed=SEED                             seed for random generator.
    -p PATIENCE, --patience PATIENCE                 min number of examples to look before stopping, default is no early stopping
    -i PATIENCE_INCR, --patience_incr=PATIENCE       wait for this much longer when a new best result is found [default: 2]
    -t IMPR_THRS, --improvement_thrs=IMPR_THRS       a relative improvemnt of this is considered significant [default: 0.995]
    -f VALID_FREQ, --validation_freq=VALID_FREQ      number of examples after which check score on dev set [default: 1000]
    -k NOISE_SAMPLES, --noise_samples=NOISE_SAMPLES  number of examples after which check score on dev set [default: 25]
"""


from dataset import Dictionary, load_corpus
from lbl import make_instances

import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging
from numpy.random import random_sample
from collections import Counter
from itertools import chain

logger = logging.getLogger(__name__)


def weighted_values(values, probabilities, size):
    """
    sample from discrete distribution
    """
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


class UnigramLanguageModel(object):
    """
    Unigram languge model
    """

    def __init__(self, train):
        """
        Initialize language model with training sentences
        """
        # count words in input sentences
        counts = Counter(chain(*train))
        total = sum((len(sentence) for sentence in train))
        words = counts.keys()
        probabilities = [float(count)/total for count in counts.values()]
        

class LogBilinearLanguageModelNCE(object):
    """
    Log-bilinear language model with noise contrastive estimation class
    """

    def __init__(self, context, V, K, context_sz, rng):
        """
        Initialize the parameters of the language model
        """
        # training contexts
        self.context = context
        # initialize context word embedding matrix R of shape (V, K)
        R_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.R = theano.shared(value=R_values, name='R', borrow=True)
        # initialize target word embedding matrix Q of shape (V, K)
        Q_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.Q = theano.shared(value=Q_values, name='Q', borrow=True)
        # initialize weight tensor C of shape (context_sz, K, K)
        C_values = np.asarray(rng.normal(0, math.sqrt(0.1), 
                                         size=(context_sz, K, K)), 
                              dtype=theano.config.floatX)
        self.C = theano.shared(value=C_values, name='C', borrow=True)
        # initialize bias vector 
        b_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(V,)), 
                              dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        # context word representations
        self.r_w = self.R[context]
        # predicted word representation for target word
        self.q_hat = T.tensordot(self.C, self.r_w, axes=[[0,1], [1,2]])
        # similarity score between predicted word and all target words
        self.s = T.transpose(T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1)))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s)
        # parameters of the model
        self.params = [self.R, self.Q, self.C, self.b]



def train_lbl(train_data, dev_data, test_data=[], 
              K=20, context_sz=2, learning_rate=1.0, 
              rate_update='simple', epochs=10, 
              batch_size=100, rng=None, patience=None, 
              patience_incr=2, improvement_thrs=0.995, 
              validation_freq=1000, noise_samples=25):
    """
    Train log-bilinear model with noise contrastive estimation
    """
    # create vocabulary from train data, plus <s>, </s>
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng

    # generate (context, target) pairs of word ids
    train_set_x, train_set_y = make_instances(train_data, vocab, context_sz)
    dev_set_x, dev_set_y = make_instances(dev_data, vocab, context_sz)
    test_set_x, test_set_y = make_instances(test_data, vocab, context_sz)

    # build the model
    logger.info("Build the model ...")
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.ivector('y')
    
    # create log-bilinear model
    lbl = LogBilinearLanguageModel(x, V, K, context_sz, rng)

    # cost function is negative log likelihood of the training data
    cost = lbl.negative_log_likelihood(y)


if __name__ == '__main__':
    import sys
    from docopt import docopt

    # parse command line arguments
    arguments = docopt(__doc__)
    log_level= logging.DEBUG if arguments['--verbose'] else logging.INFO
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)
    logger.setLevel(log_level)
    word_dim = int(arguments['--word_dim'])
    context_sz = int(arguments['--context_sz'])
    learn_rate = float(arguments['--learn_rate'])
    rate_update = arguments['--rate_update']
    epochs = int(arguments['--epochs'])
    batch_sz = int(arguments['--batch_size'])
    seed = int(arguments['--seed']) if arguments['--seed'] else None
    patience = int(arguments['--patience']) if arguments['--patience'] else None
    patience_incr = int(arguments['--patience_incr'])
    improvement_thrs = float(arguments['--improvement_thrs'])
    validation_freq = int(arguments['--validation_freq'])
    noise_samples = int(arguments['--noise_samples'])

    # load data
    logger.info("Load data ...")
    with open(arguments['<train_data>'], 'rb') as fin:
        train_data = [line.split() for line in fin.readlines() if line.strip()]    
    with open(arguments['<dev_data>'], 'rb') as fin:
        dev_data = [line.split() for line in fin.readlines() if line.strip()]
    if arguments['<test_data>']:
        with open(arguments['<test_data>'], 'rb') as fin:
            test_data = [line.split() for line in fin.readlines() if line.strip()]
    else:
        test_data = []

    # create random number generator
    rng_state = np.random.RandomState(seed)

    # train lm model
    train_lbl(train_data, dev_data, test_data=test_data, 
              K=word_dim, context_sz=context_sz, learning_rate=learn_rate, 
              rate_update=rate_update, epochs=epochs, batch_size = batch_sz, 
              rng=rng_state, patience=patience, patience_incr=patience_incr, 
              improvement_thrs=improvement_thrs, validation_freq=validation_freq,
              noise_sample=noise_samples)
              

