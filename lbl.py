"""

The lob-bilinear language model from (Mnih and Teh, ICML 2012)

References:
 A fast and simple algorithm for training neural probabilistic language models. Andriy Mnih and Yee Whye Teh.
 International Conference on Machine Learning 2012 (ICML 2012) 

Usage: lbl.py [-v] [-k WORD_DIM] [-n CONTEXT_SZ] [-l LEARN_RATE] [-e EPOCHS] [-b BATCH_SZ] [-s SEED] TRAIN_DATA [DEV_DATA] [TEST_DATA]

Arguments:
 TRAIN_DATA       training data of tokenized text, one sentence per line.
 DEV_DATA         development data of tokenized text, one sentence per line.
 TEST_DATA        test data of tokenized text, one sentence per line.

Options:
    -v, --verbose                            Print debug information
    -k WORD_DIM, --word_dim=WORD_DIM         dimension of word embeddings, default 20.
    -n CONTEXT_SZ, --context_sz=CONTEXT_SZ   size of n-gram context window, default 2.
    -l LEARN_RATE, --learn_rate=LEARN_RATE   learning rate parameter, default 0.1.
    -e EPOCHS, --epochs=EPOCHS               number of training epochs, default 10.
    -b BATCH_SIZE, --batch_size=BATCH_SIZE   size of mini-batch for training, default 100.
    -s SEED, --seed=SEED                     seed for random generator.

"""

from dataset import Dictionary, load_corpus

import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging


# Usage: lbl.py [-v] [-k WORD_DIM] [-n CONTEXT_SZ] [-l LEARN_RATE] [-e EPOCHS] [-b BATCH_SZ] [-s SEED] TRAIN_DATA [DEV_DATA] [TEST_DATA]

logger = logging.getLogger(__name__)


class LogBilinearLanguageModel(object):
    """
    Log-bilinear language model class
    """

    def __init__(self, context, V, K, context_sz, rng):
        """
        Initialize the parameters of the language model
        """

        # training contexts
        self.context = context

        # initialize context word embedding matrix R of shape (V, K)
        R_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), dtype=theano.config.floatX)
        self.R = theano.shared(value=R_values, name='R', borrow=True)

        # initialize target word embedding matrix Q of shape (V, K)
        Q_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), dtype=theano.config.floatX)
        self.Q = theano.shared(value=Q_values, name='Q', borrow=True)

        # initialize weight tensor C of shape (context_sz, K, K)
        C_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(context_sz, K, K)), dtype=theano.config.floatX)
        self.C = theano.shared(value=C_values, name='C', borrow=True)

        # initialize bias vector 
        b_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(1,V)), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

         # r_w : context word representations
        self.r_w = self.R[context]
        # q_hat : predicted word representation for target word
        self.q_hat = T.tensordot(self.C, self.r_w, axes=[[0,1], [1,2]])
        # s_wh : similarity score between predicted word and all target words
        self.s = T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s)

        # parameters of the model
        self.params = [self.R, self.Q, self.C, self.b]


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(T.transpose(self.p_w_given_h))[T.arange(y.shape[0]),y])

    def perplexity(self, y):
        return T.pow(T.prod(T.inv(T.transpose(self.p_w_given_h)[T.arange(y.shape[0]),y])), 1.0/y.shape[0])

        
def generate_instances(corpus, vocab, context_sz):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=borrow)
        return shared_x, shared_y
    data = []
    labels = []        
    for sentence in corpus:
        # add 'start of sentence' / 'end of sentence' context
        sentence = ['<s>'] * context_sz + sentence + ['</s>'] * context_sz
        sentence = vocab.doc_words_to_ids(sentence, update_dict=False, unk='<unk>')
        for instance in zip(*(sentence[i:] for i in xrange(context_sz+1))):
            data.append(instance[:-1])
            labels.append(instance[-1])

    train_set_x, train_set_y = shared_dataset([data, labels])
    return train_set_x, train_set_y


def train_lbl(train_data, dev_data = [], test_data = [], 
              K = 20, context_sz = 2, learning_rate=0.1, 
              epochs=10, batch_size=100, rng = None):
    """
    Train log-bilinear model
    """
    # create vocabulary from train data, plus <s>, </s>
    vocab = Dictionary.from_corpus(train_data)
    vocab.lookup_id('<s>', update_dict = True)
    vocab.lookup_id('</s>', update_dict = True)
    V = vocab.size()

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng

    # generate (context, target) pairs of word ids
    train_set_x, train_set_y = generate_instances(train_data, vocab, context_sz)
    dev_set_x, dev_set_y = generate_instances(dev_data, vocab, context_sz)
    test_set_x, test_set_y = generate_instances(test_data, vocab, context_sz)

    # number of minibatches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    logger.info("Build the model ...")
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.ivector('y')
    
    # create log-bilinear model
    lbl = LogBilinearLanguageModel(x, V, K, context_sz, rng)

    # cost function during training is negative log likelihood of the training data
    cost = lbl.negative_log_likelihood(y)

    # perplexity of the data given the model
    ppl = lbl.perplexity(y)

    # compute the gradient
    gparams = []
    for param in lbl.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameter of the model
    updates = []
    for param, gparam in zip(lbl.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling theano functions that computes perplexity on the dev and test set
    ppl_model_dev = theano.function(inputs=[index], outputs=ppl,
                                 givens={x: dev_set_x[index * batch_size:(index+1) * batch_size],
                                         y: dev_set_y[index * batch_size:(index+1) * batch_size]
                                         })

    ppl_model_test = theano.function(inputs=[index], outputs=ppl,
                                 givens={x: test_set_x[index * batch_size:(index+1) * batch_size],
                                         y: test_set_y[index * batch_size:(index+1) * batch_size]
                                         })
    
    # compiling theano function 'train model' that returns the cost
    # and updates the parameter of the model 
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x[index * batch_size:(index+1) * batch_size],
                                          y: train_set_y[index * batch_size:(index+1) * batch_size]
                                          })

            
    # train model
    logger.info("training model...")
    patience = 5000 
    patience_increase = 2
    improvement_threshold = 0.995
    validation_freq = min(n_train_batches, patience/2)
    best_params = None
    best_dev_ppl = np.inf
    test_ppl = np.inf
    test_core = 0
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < epochs) and (not done_looping):
        logger.debug('epoch %i' % epoch) 
        for minibatch_index in xrange(n_train_batches):
            if minibatch_index > 0 and minibatch_index % 100 == 0:
                logger.debug('minibatch %i' % minibatch_index) 
            minibatch_avg_cost  = train_model(minibatch_index)
            # iteration number
            itr = epoch * n_train_batches + minibatch_index            
            if itr % validation_freq == 0:
                # compute perplexity on dev set
                dev_ppl = np.mean([ppl_model_dev(i) for i in xrange(n_dev_batches)])
                logger.debug('epoch %i, minibatch %i/%i, dev ppl %.4f' % (epoch, minibatch_index+1,
                                                                                   n_train_batches, dev_ppl))

                # if we got the lowest perplexity until now
                if dev_ppl < best_dev_ppl:
                    # improve patience if loss improvement is good enough
                    if dev_ppl < best_dev_ppl * improvement_threshold:
                        patience = max(patience, itr * patience_increase)
                    best_dev_ppl = dev_ppl
                    # test perplexity
                    test_ppl = np.mean([ppl_model_test(i) for i in xrange(n_test_batches)])
                    logger.debug('  epoch %i, minibatch %i/%i, test ppl %.4f' % (epoch, minibatch_index+1, 
                                                                                  n_train_batches, test_ppl))
                    
            if patience <= itr:
                done_looping = True
                break
        # increment epoch
        epoch = epoch + 1

    end_time = time.clock()
    total_time = end_time - start_time
    logger.info('Optimization complete with best dev ppl of %.4f and test ppl %.4f' % (best_dev_ppl, test_ppl))
    logger.info('Training took %d epochs, with %.1f epochs/sec' % (epoch+1, 
                float(epoch+1) / total_time))
    logger.info("Total training time %d days %d hours %d min %d sec." % (total_time/60/60/24, total_time/60/60%24, total_time/60%60, total_time%60))

    
if __name__ == '__main__':
    import sys
    from docopt import docopt

    # parse command line arguments
    arguments = docopt(__doc__)
    log_level= logging.DEBUG if arguments['--verbose'] else logging.ERROR
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)
    logger.setLevel(log_level)
    word_dim = arguments['--word_dim'] or 20
    context_sz = arguments['--context_sz'] or 2
    learn_rate = arguments['--learn_rate'] or 0.1
    epochs = arguments['--epochs'] or 10
    batch_sz = arguments['--batch_size'] or 100
    seed = arguments['--seed'] or None

    logger.info("Load data ...")
    with open(arguments['TRAIN_DATA'], 'rb') as fin:
        train_data = [line.split() for line in fin.readlines() if line.strip()]
    
    if arguments['DEV_DATA']:
        with open(arguments['DEV_DATA'], 'rb') as fin:
            dev_data = [line.split() for line in fin.readlines() if line.strip()]
    else:
        dev_data = []

    if arguments['TEST_DATA']:
        with open(arguments['TEST_DATA'], 'rb') as fin:
            test_data = [line.split() for line in fin.readlines() if line.strip()]
    else:
        test_data = []

    # random generator
    rng_state = np.random.RandomState(seed)

    # train lm model
    train_lbl(train_data, dev_data=dev_data, test_data = test_data, K=word_dim, context_sz=context_sz, learning_rate=learn_rate, epochs=epochs, batch_size = batch_sz, rng=rng_state)

