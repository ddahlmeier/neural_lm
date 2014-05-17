"""

The log-bilinear language model with noise constrastive estimation 
from (Mnih and Teh, ICML 2012)

References:
 A fast and simple algorithm for training neural probabilistic language models. 
 Andriy Mnih and Yee Whye Teh.
 International Conference on Machine Learning 2012 (ICML 2012) 

Usage: lbl_nce.py [--verbose] [--word_dim WORD_DIM] [--context_sz CONTEXT_SZ] 
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
    -p PATIENCE, --patience PATIENCE                 min number of examples to look at, default is no early stopping
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
from collections import Counter, Iterable
from itertools import chain

logger = logging.getLogger(__name__)


class UnigramLanguageModel(object):
    """
    Unigram language model
    """

    def __init__(self, train, vocab):
        """
        Initialize language model with training sentences
        """
        probabilities_values = np.zeros(vocab.size())
        total = 0
        for word in chain(*train):
            word_id = vocab.lookup_id(word, update_dict=False)
            if not word_id:
                raise ValueError("Training data contained word not in the vocabulary: %s" % word)            
            probabilities_values[word_id] += 1
            total += 1
        probabilities_values /= total
        self.probabilities = theano.shared(probabilities_values, borrow=True)
        self.bins = np.add.accumulate(probabilities_values)

    def likelihood(self, word_ids):
        """
        probability of a tensor of word is under the unigram model.
        """
        return self.probabilities[word_ids]

    def samples(self, size):
        """
        sample of size 'size' from unigram language model
        """
        return np.digitize(random_sample(size), self.bins)



class LogBilinearLanguageModelNCE(object):
    """
    Log-bilinear language model with noise contrastive estimation class
    """

    def __init__(self, context, V, K, context_sz, rng):
        """
        Initialize the parameters of the language model
        """
        # vocabulary size
        self.V = V
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
        # similarity between predictions and all target words
        # TODO: do we need to compute similarity to all other words? does this destroy the benefit of NCE?
        self.s = T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1))
        # normalized model score for test time
        self.p_w_given_h = T.nnet.softmax(self.s)
        # parameters of the model
        self.params = [self.R, self.Q, self.C, self.b]

    def unnormalized_likelihood(self, y, c=1.0):
        """
        compute unnormalized likelihood of target words y
        """
        return self.s * T.exp(c)

    def unnormalized_neg_log_likelihood(self, y, c=1.0):
        """
        compute unnormalized log-likelihood of target words y
        """
        return -T.mean(T.log2(self.s * T.exp(c))[y, T.arange(y.shape[0])])

    def negative_log_likelihood(self, y):
        """
        compute negative log-likelihood of target words y
        explicitly normalize predicted scores
        """
        return -T.mean(T.log2(self.p_w_given_h)[y, T.arange(y.shape[0])])

        
        

def train_lbl(train_data, dev_data, test_data=[], 
              K=20, context_sz=2, learning_rate=1.0, 
              rate_update='simple', epochs=10, 
              batch_size=100, rng=None, patience=None, 
              patience_incr=2, improvement_thrs=0.995, 
              validation_freq=1000, noise_data_ratio=25):
    """
    Train log-bilinear model with noise contrastive estimation
    """
    # create vocabulary from train data, plus <s>, </s>
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()
    print vocab.vocab
    logger.debug("Vocabulary size: %d" % V)

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng

    # generate (context, target) pairs of word ids
    train_set_x, train_set_y = make_instances(train_data, vocab, context_sz)
    dev_set_x, dev_set_y = make_instances(dev_data, vocab, context_sz)
    test_set_x, test_set_y = make_instances(test_data, vocab, context_sz)

    # generate noise samples
    noise_model = UnigramLanguageModel(train_data, vocab)
    data_sz = train_set_x.shape.eval()[0]
    noise_set = theano.shared(np.asarray(noise_model.samples(noise_data_ratio * data_sz), 
                                          dtype=np.int32), borrow=True)

    # number of minibatches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    logger.info("Build the model ...")
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.ivector('y')
    noise = T.ivector('noise')
    
    # create log-bilinear model
    lbl = LogBilinearLanguageModelNCE(x, V, K, context_sz, rng)

    # cost function is the unnormalized log-probability
    cost = lbl.unnormalized_neg_log_likelihood(y)
    noise_cost = lbl.unnormalized_neg_log_likelihood(noise)
    cost_normalized = lbl.negative_log_likelihood(y)

    # compute gradient 
    gparams = []
    noise_gparams = []
    for param in lbl.params:
        gparam = T.grad(cost, param)
        noise_gparam = T.grad(noise_cost, param)
        gparams.append(gparam)
        noise_gparams.append(noise_gparam)

    # specify NCE objective update step for model parameter
    updates = []
    for param, gparam, noise_gparam in zip(lbl.params, gparams, noise_gparams):
        # k * P_n(w) / (P_h(w) + k * P_n(w))
        nce_weight = noise_data_ratio * noise_model.likelihood(y) / (lbl.unnormalized_neg_log_likelihood(y) + noise_data_ratio*noise_model.likelihood(y))
        # nce update
        # update = nce_weight*gparam
        update = gparam
        # debug: just add half of the update
        # updates.append((param, param-learning_rate*update))

        # gradient approximation with noise samples
        # P_h(w) / (P_h(w) + k * P_n(w))
        # noise_weight = lbl.unnormalized_neg_log_likelihood(noise) / (lbl.unnormalized_neg_log_likelihood(noise) + noise_data_ratio*noise_model.likelihood(noise))
        # noise_update = noise_weight*noise_gparam
        noise_update = noise_gparam
        # # sum over k noise samples
        noise_update.reshape((noise_data_ratio, y.shape[0])).sum(axis=0)
        # noise_update.reshape((noise_data_ratio, y.shape[0])).sum(axis=0)
        # # overall update step on objective function J
        updates.append((param, param-learning_rate*(update-noise_update)))
        

    # function that computes normalized log-probability of the dev set
    logprob_dev = theano.function(inputs=[index], outputs=cost_normalized,
                                  givens={x: dev_set_x[index*batch_size:
                                                           (index+1)*batch_size],
                                          y: dev_set_y[index*batch_size:
                                                           (index+1)*batch_size]
                                          })


    # function that computes normalized log-probability of the test set
    logprob_test = theano.function(inputs=[index], outputs=cost_normalized,
                                   givens={x: test_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                           y: test_set_y[index*batch_size:
                                                             (index+1)*batch_size]
                                           })
    
    # function that returns the unnormalized cost and updates the parameter 
    # debug
    # return udpate for first paramter (R matrix)
    # train_model = theano.function(inputs=[index], outputs=nce_weight,
    #                               updates=updates,
    #                               givens={x: train_set_x[index*batch_size:
    #                                                          (index+1)*batch_size],
    #                                       y: train_set_y[index*batch_size:
    #                                                          (index+1)*batch_size],
    #                                       noise: noise_set[index*batch_size*noise_data_ratio:
    #                                                            (index+1)*batch_size*noise_data_ratio]
    #                                       },
    #                               on_unused_input='warn'
    #                               )

    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                          y: train_set_y[index*batch_size:
                                                             (index+1)*batch_size],
                                          noise: noise_set[index*batch_size*noise_data_ratio:
                                                               (index+1)*batch_size*noise_data_ratio]
                                          },
                                  on_unused_input='warn'
                                  )

    # train_model = theano.function(inputs=[index], outputs=cost,
    #                               givens={x: train_set_x[index*batch_size:
    #                                                          (index+1)*batch_size],
    #                                       y: train_set_y[index*batch_size:
    #                                                          (index+1)*batch_size],
    #                                      })
        
    # perplexity functions
    def compute_dev_logp():
        return np.mean([logprob_dev(i) for i in xrange(n_dev_batches)])

    def compute_test_logp():
        return np.mean([logprob_test(i) for i in xrange(n_test_batches)])

    def ppl(neg_logp):
        return np.power(2.0, neg_logp)

    # train model
    logger.info("training model...")
    best_params = None
    last_epoch_dev_ppl = np.inf
    best_dev_ppl = np.inf
    test_ppl = np.inf
    test_core = 0
    start_time = time.clock()
    done_looping = False

    for epoch in xrange(epochs):
        if done_looping:
            break
        logger.debug('epoch %i' % epoch) 
        for minibatch_index in xrange(n_train_batches):
            itr = epoch * n_train_batches + minibatch_index
            # tmp = train_model(minibatch_index)
            # print "shape tmp:", tmp.shape
            train_logp = train_model(minibatch_index)
            logger.debug('epoch %i, minibatch %i/%i, train minibatch log prob %.4f ppl %.4f' % 
                         (epoch, minibatch_index+1, n_train_batches, 
                          train_logp, ppl(train_logp)))
            if (itr+1) % validation_freq == 0:
                # compute perplexity on dev set, lower is better
                dev_logp = compute_dev_logp()
                dev_ppl = ppl(dev_logp)
                logger.debug('epoch %i, minibatch %i/%i, dev log prob %.4f ppl %.4f' % 
                             (epoch, minibatch_index+1, n_train_batches, 
                              dev_logp, ppl(dev_logp)))
                # if we got the lowest perplexity until now
                if dev_ppl < best_dev_ppl:
                    # improve patience if loss improvement is good enough
                    if patience and dev_ppl < best_dev_ppl * improvement_thrs:
                        patience = max(patience, itr * patience_incr)
                    best_dev_ppl = dev_ppl
                    test_logp = compute_test_logp()
                    test_ppl = ppl(test_logp)
                    logger.debug('epoch %i, minibatch %i/%i, test log prob %.4f ppl %.4f' % 
                                 (epoch, minibatch_index+1, n_train_batches, 
                                  test_logp, ppl(test_logp)))
            # stop learning if no improvement was seen for a long time
            if patience and patience <= itr:
                done_looping = True
                break
        # adapt learning rate
        if rate_update == 'simple':
            # set learning rate to 1 / (epoch+1)
            learning_rate = 1.0 / (epoch+1)
        elif rate_update == 'adaptive':
            # half learning rate if perplexity increased at end of epoch (Mnih and Teh 2012)
            this_epoch_dev_ppl = ppl(compute_dev_logp())
            if this_epoch_dev_ppl > last_epoch_dev_ppl:
                learning_rate /= 2.0
            last_epoch_dev_ppl = this_epoch_dev_ppl
        elif rate_update == 'constant':
            # keep learning rate constant
            pass
        else:
            raise ValueError("Unknown learning rate update strategy: %s" %rate_update)
        
    end_time = time.clock()
    total_time = end_time - start_time
    logger.info('Optimization complete with best dev ppl of %.4f and test ppl %.4f' % 
                (best_dev_ppl, test_ppl))
    logger.info('Training took %d epochs, with %.1f epochs/sec' % (epoch+1, 
                float(epoch+1) / total_time))
    logger.info("Total training time %d days %d hours %d min %d sec." % 
                (total_time/60/60/24, total_time/60/60%24, total_time/60%60, total_time%60))
    # return model
    return lbl

    

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
    lbl = train_lbl(train_data, dev_data, test_data=test_data, 
              K=word_dim, context_sz=context_sz, learning_rate=learn_rate, 
              rate_update=rate_update, epochs=epochs, batch_size = batch_sz, 
              rng=rng_state, patience=patience, patience_incr=patience_incr, 
              improvement_thrs=improvement_thrs, validation_freq=validation_freq,
              noise_data_ratio=noise_samples)

              
