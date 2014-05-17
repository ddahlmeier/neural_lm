"""

The hybrid log-bilinear language model (Mnih and Teh, ICML 2012) on word and character representations


Usage: lbl_hybrid.py [--verbose] [--word_dim WORD_DIM] 
              [--word_context_sz WORD_CONTEXT_SZ] [--char_context_sz CHAR_CONTEXT_SZ]
              [--learn_rate LEARN_RATE] [--rate_update RATE_UPDATE] 
              [--epochs EPOCHS] [--batch_size BATCH_SZ] [--seed SEED]  
              [--patience PATIENCE] [--patience_incr PATIENCE_INCR] 
              [--improvement_thrs IMPR_THRS] [--validation_freq VALID_FREQ] 
              [--model MODEL_FILE] 
              <train_data> <dev_data> [<test_data>]

Arguments:
 train_data       training data of tokenized text, one sentence per line.
 dev_data         development data of tokenized text, one sentence per line.
 test_data        test data of tokenized text, one sentence per line.

Options:
    -v, --verbose                                          Print debug information
    -k WORD_DIM, --word_dim=WORD_DIM                       dimension of word embeddings [default: 100]
    -n WORD_CONTEXT_SZ, --word_context_sz=WORD_CONTEXT_SZ  size of word n-gram context window [default: 2]
    -m CHAR_CONTEXT_SZ, --char_context_sz=CHAR_CONTEXT_SZ  size of character n-gram context window [default: 2]
    -l LEARN_RATE, --learn_rate=LEARN_RATE                 initial learning rate parameter [default: 1]
    -u RATE_UPDATE, --rate_update=RATE_UPDATE              learning rate update: 'simple', 'adaptive' [default: simple]
    -e EPOCHS, --epochs=EPOCHS                             number of training epochs [default: 10]
    -b BATCH_SIZE, --batch_size=BATCH_SIZE                 size of mini-batch for training [default: 100]
    -s SEED, --seed=SEED                                   seed for random generator.
    -p PATIENCE, --patience PATIENCE                       min number of examples to look before stopping, default is no early stopping
    -i PATIENCE_INCR, --patience_incr=PATIENCE             wait for this much longer when a new best result is found [default: 2]
    -t IMPR_THRS, --improvement_thrs=IMPR_THRS             a relative improvemnt of this is considered significant [default: 0.995]
    -f VALID_FREQ, --validation_freq=VALID_FREQ            number of examples after which check score on dev set [default: 1000]
    -m MODEL_FILE, --model=MODEL_FILE                      file to save the model to. Default none.
"""

from dataset import Dictionary, load_corpus

import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging

logger = logging.getLogger(__name__)


class LogBilinearLanguageModel(object):
    """
    Log-bilinear language model class
    """

    def __init__(self, word_context, char_context, V, K, word_context_sz, char_context_sz, rng):
        """
        Initialize the parameters of the language model
        """
        # word training contexts
        self.word_context = word_context
        # character training contexts
        self.char_context = char_context

        # initialize context word embedding matrix Rw of shape (V, K)
        Rw_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.Rw = theano.shared(value=Rw_values, name='Rw', borrow=True)
        # initialize context character embedding matrix Rc of shape (V, K)
        Rc_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.Rc = theano.shared(value=Rc_values, name='Rc', borrow=True)

        # initialize target word embedding matrix Q of shape (V, K)
        Q_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.Q = theano.shared(value=Q_values, name='Q', borrow=True)
        # initialize word weight tensor Cw of shape (word_context_sz, K, K)
        Cw_values = np.asarray(rng.normal(0, math.sqrt(0.1), 
                                          size=(word_context_sz, K, K)), 
                              dtype=theano.config.floatX)
        self.Cw = theano.shared(value=Cw_values, name='Cw', borrow=True)
        # initialize character weight tensor Cc of shape (char_context_sz, K, K)
        Cc_values = np.asarray(rng.normal(0, math.sqrt(0.1), 
                                          size=(char_context_sz, K, K)), 
                               dtype=theano.config.floatX)
        self.Cc = theano.shared(value=Cc_values, name='Cc', borrow=True)
        # initialize bias vector 
        b_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(V,)), 
                              dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        # context word representations
        self.r_w = self.Rw[word_context]
        # context character representations
        self.r_c = self.Rc[char_context]
        # predicted word representation for target word by word context
        self.qw_hat = T.tensordot(self.Cw, self.r_w, axes=[[0,1], [1,2]])
        # predicted word representation for target word by character context
        self.qc_hat = T.tensordot(self.Cc, self.r_c, axes=[[0,1], [1,2]])
        # combine word and charafter predictions
        self.q_hat = self.qw_hat + self.qc_hat
        # similarity score between predicted word and all target words
        self.s = T.transpose(T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1)))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s)
        # parameters of the model
        self.params = [self.Rw, self.Rc, self.Q, self.Cw, self.Cc, self.b]

    def negative_log_likelihood(self, y):
        # take the logarithm with base 2
        return -T.mean(T.log2(self.p_w_given_h)[T.arange(y.shape[0]), y])

        
def make_instances(corpus, vocab, word_context_sz, char_context_sz, start_symb='<s>', end_symb='</s>'):
    def shared_dataset(data_xy, borrow=True):
        data_word_x, data_char_x, data_y = data_xy
        shared_word_x = theano.shared(np.asarray(data_word_x, dtype=np.int32), borrow=borrow)
        shared_char_x = theano.shared(np.asarray(data_char_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=borrow)
        return shared_word_x, shared_char_x, shared_y
    data_words = []
    data_char = []
    labels = []        
    for sentence in corpus:
        # add 'start of sentence' and 'end of sentence' context
        print ' '.join(sentence).encode('utf-8')
        sentence = [start_symb] * word_context_sz + sentence + [end_symb] * word_context_sz
        for instance in zip(*(sentence[i:] for i in xrange(word_context_sz+1))):
            context = instance[:-1]
            char_context = list(filter(lambda w: w!= ' ', ''.join(context)))[-char_context_sz:]
            # TODO: check if updating vocabulary here is ok, probably not because of matrix dimensions
            data_words.append(vocab.doc_words_to_ids(context, update_dict=True))
            data_char.append(vocab.doc_words_to_ids(char_context, update_dict=True))
            print "word context:", ' '.join(context).encode('utf-8')
            print "char context:", ' '.join(char_context).encode('utf-8')
            labels.append(instance[-1])
    train_word_x, train_char_x, train_set_y = shared_dataset([data_words, data_char, labels])
    return train_word_x, train_char_x, train_set_y


def train_lbl(train_data, dev_data, test_data=[], 
              K=20, word_context_sz=2, char_context_sz=2,
              learning_rate=1.0, rate_update='simple', 
              epochs=10, batch_size=100, rng=None, 
              patience=None, patience_incr=2, 
              improvement_thrs=0.995, validation_freq=1000):
    """
    Train log-bilinear model
    """
    # create vocabulary from train data, plus <s>, </s>
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng

    # generate (context, target) pairs of word ids
    train_word_x, train_char_x, train_set_y = make_instances(train_data, vocab, word_context_sz, char_context_sz)
    dev_word_x, dev_char_x, dev_set_y = make_instances(dev_data, vocab, word_context_sz, char_context_sz)
    test_word_x, test_char_x, test_set_y = make_instances(test_data, vocab, word_context_sz, char_context_sz)

    # number of minibatches for training
    n_train_batches = train_word_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_word_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_word_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    logger.info("Build the model ...")
    index = T.lscalar()
    x_word = T.imatrix('x_word')
    x_char = T.imatrix('x_char')
    y = T.ivector('y')
    
    # create log-bilinear model
    lbl = LogBilinearLanguageModel(x_word, x_char, V, K, word_context_sz, char_context_sz, rng)

    # cost function is negative log likelihood of the training data
    cost = lbl.negative_log_likelihood(y)

    # compute the gradient
    gparams = []
    for param in lbl.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameter of the model
    updates = []
    for param, gparam in zip(lbl.params, gparams):
        updates.append((param, param-learning_rate*gparam))

    # function that computes log-probability of the dev set
    logprob_dev = theano.function(inputs=[index], outputs=cost,
                                  givens={x_word: dev_word_x[index*batch_size:
                                                                 (index+1)*batch_size],
                                          x_char: dev_char_x[index*batch_size:
                                                                 (index+1)*batch_size],
                                          y: dev_set_y[index*batch_size:
                                                           (index+1)*batch_size]
                                          })


    # function that computes log-probability of the test set
    logprob_test = theano.function(inputs=[index], outputs=cost,
                                   givens={x_word: test_word_x[index*batch_size:
                                                                   (index+1)*batch_size],
                                           x_char: test_char_x[index*batch_size:
                                                                   (index+1)*batch_size],
                                           y: test_set_y[index*batch_size:
                                                             (index+1)*batch_size]
                                           })
    
    # function that returns the cost and updates the parameter 
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={x_word: train_word_x[index*batch_size:
                                                                   (index+1)*batch_size],
                                          x_char: train_char_x[index*batch_size:
                                                                   (index+1)*batch_size],
                                          y: train_set_y[index*batch_size:
                                                             (index+1)*batch_size]
                                          })

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
    word_context_sz = int(arguments['--word_context_sz'])
    char_context_sz = int(arguments['--char_context_sz'])
    learn_rate = float(arguments['--learn_rate'])
    rate_update = arguments['--rate_update']
    epochs = int(arguments['--epochs'])
    batch_sz = int(arguments['--batch_size'])
    seed = int(arguments['--seed']) if arguments['--seed'] else None
    patience = int(arguments['--patience']) if arguments['--patience'] else None
    patience_incr = int(arguments['--patience_incr'])
    improvement_thrs = float(arguments['--improvement_thrs'])
    validation_freq = int(arguments['--validation_freq'])
    outfile = arguments['--model']

    # load data
    logger.info("Load data ...")
    with open(arguments['<train_data>'], 'rb') as fin:
        train_data = [line.decode('utf-8').split() for line in fin.readlines() if line.strip()]    
    with open(arguments['<dev_data>'], 'rb') as fin:
        dev_data = [line.decode('utf-8').split() for line in fin.readlines() if line.strip()]
    if arguments['<test_data>']:
        with open(arguments['<test_data>'], 'rb') as fin:
            test_data = [line.decode('utf-8').split() for line in fin.readlines() if line.strip()]
    else:
        test_data = []

    # create random number generator
    rng_state = np.random.RandomState(seed)

    # train lm model
    lbl = train_lbl(train_data, dev_data, test_data=test_data, 
                    K=word_dim, word_context_sz=word_context_sz, 
                    char_context_sz=char_context_sz, learning_rate=learn_rate, 
                    rate_update=rate_update, epochs=epochs, batch_size = batch_sz, 
                    rng=rng_state, patience=patience, patience_incr=patience_incr, 
                    improvement_thrs=improvement_thrs, validation_freq=validation_freq)
    # save the model
    if outfile:
        logger.info("Saving model ...")
        fout = open(outfile, 'wb')
        for param in lbl.params:
            cPickle.dump(param.get_value(borrow=True), fout, -1)
        fout.close()
        
        

