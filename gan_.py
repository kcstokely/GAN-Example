############################################################################################################################################    

'''
    This is code meant as an illustration on setting up a Generative Adversarial Network.
    
    Both Generator and Discriminator are set up as deep feed-forwards networks (MLP's).

    We use MNIST just for show.  One really shouldn't be using MLP's for image tasks.
'''
    
import os
import sys
import gzip
import cPickle
import datetime
import PIL.Image as Image

import numpy
import theano
import theano.tensor as T

from foxhound import updates

theano.config.floatX = 'float32'

############################################################################################################################################    
    
class HiddenLayer(object):

    def __init__(self, rng, n_vis, n_hid, W=None, h=None):
        
        if W is None:
            W_vals = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (n_vis + n_hid)),
                    high = numpy.sqrt(6. / (n_vis + n_hid)),
                    size = (n_vis, n_hid)
                ),
                dtype = theano.config.floatX
            )
            W = theano.shared(value = W_vals, borrow = True)
            
        if h is None:
            h_vals = numpy.zeros((n_hid,), dtype=theano.config.floatX)
            h = theano.shared(value = h_vals, borrow = True)
            
        self.W = W
        self.h = h
        
    def lin_out(self, inp):
        return T.dot(inp, self.W) + self.h
        
    def act_out(self, inp):
        return T.nnet.sigmoid(self.lin_out(inp))

############################################################################################################################################    

class MLP(object):

    def __init__(self, rng, nodes, W=None, h=None):
        
        self.num_layers = len(nodes)-1
        
        if W is None:
            W = [None] * self.num_layers
        if h is None:
            h = [None] * self.num_layers        
        
        self.W = []
        self.h = []
        self.hid = []
        for n in range(self.num_layers):
            self.hid.append(HiddenLayer(rng, nodes[n], nodes[n+1], W=W[n], h=h[n]))
            self.W.append(self.hid[n].W)
            self.h.append(self.hid[n].h)
            
        self.params = self.W + self.h
        
    def lin_out(self, inp):
        out = self.hid[0].act_out(inp)
        for n in range(1, self.num_layers-1):
            out = self.hid[n].act_out(out)
        out = self.hid[-1].lin_out(out)
        return out
        
    def act_out(self, inp):
        return T.nnet.sigmoid(self.lin_out(inp))

############################################################################################################################################    

if __name__ == '__main__':

    #########
    #########   PARAMS
    ######### 

    # random number generation
    dis_rng = numpy.random.RandomState()
    gen_rng = numpy.random.RandomState()

    # number of nodes in x-vector (image)
    n_x = 28*28
    
    # nuber of nodes in z-vector
    n_z = 100
    
    # node structure of discriminator
    dis_nodes = [n_x, 500, 500, 1]
    
    # node structure of generator
    gen_nodes = [n_z, 500, 1000, n_x]
    
    # training parameters
    smooth = 0.9
    batch_size = 50
    learning_rate_dis = 0.0001
    learning_rate_gen = 0.001

    # training protocol
    start = 1
    num_epochs = 1
    
    #########
    #########   CREATE DISCRIMINATOR AND GENERATOR   
    ######### 
    
    '''
        If 'param_.pkl' exists, weights are loaded, else they are initialized randomly.
    '''
    
    try:
        fp = open('param_.pkl', 'rb')
        gen_par = cPickle.load(fp)
        dis_par = cPickle.load(fp)
        fp.close()
        gen_num = len(gen_par)/2
        dis_num = len(dis_par)/2
        generator     = MLP(gen_rng, gen_nodes, W=gen_par[:gen_num], h=gen_par[gen_num:])
        discriminator = MLP(dis_rng, dis_nodes, W=dis_par[:dis_num], h=dis_par[dis_num:])
        print '### GAN LOADED ###'
    except:
        generator     = MLP(gen_rng, gen_nodes)
        discriminator = MLP(dis_rng, dis_nodes)
        print '### GAN CREATED ###'

    #########
    #########   CREATE TRAINING FUNCTIONS   
    #########
        
    '''
        These are symbolic variables used to create the computational graphs:
            x represents that which we are generating
            y represents the labels indicating whether an image is data or generated
            z represents vector used as input to the generator
    '''

    x = T.matrix('x')
    y = T.matrix('y')
    z = T.matrix('z')

    '''
        The discriminator will be set up to minimize the cross-entropy between the
            output and the expectation that the input is real data.
        
        The generator will be set up to minimize the cross entropy between the
            discriminator's output on a generated sample and the expectation that
            this generated sample is generated (converse to above, and obvs. unity).
    
        The generator is trained on random z-vectors.
        
        The discriminator is trained on both real data and the x-vectors generated
            from the z-vectors which the generator was trined on.
    
        There are two ways we might proceed:
        
            A)  create a function which trains the discriminator on any x-vector,
                    which can then be called on real data or generated images.
                    
            B)  create two functions for training the discriminator: one which takes
                    an x-vector of data, and one which takes a z-vector, uses the
                    generator to create an x-vector from that, then feeds it to the
                    discrimintor.
                    
        In essence, these are identical, but in practice evaluating the generator to
            turn z-vectors into x-vectors to feed the discriminator training function
            (method A) is very slow, hence method B is used below.
            
        We use Adam to intelligently choose our gradient descent stepsize.

        We use smoothing, ie when training the discriminator on real data, we aim for
            a probability of (s = 1 - epsilon ) instead of unity. 
    '''
    
    dis_cost_data = T.nnet.binary_crossentropy(discriminator.act_out(x), y).mean()
    
    dis_updater_data = updates.Adam(lr=learning_rate_dis)
    dis_updates_data = dis_updater_data(discriminator.params, dis_cost_data)
    
    train_dis_data = theano.function(inputs  = [x, y], outputs = [], updates = dis_updates_data)
    
    dis_cost_fake = T.nnet.binary_crossentropy(discriminator.act_out(generator.act_out(z)), y).mean()
    
    dis_updater_fake = updates.Adam(lr=learning_rate_dis)
    dis_updates_fake = dis_updater_fake(discriminator.params, dis_cost_fake)

    train_dis_fake = theano.function(inputs  = [z, y], outputs = [], updates = dis_updates_fake)
    
    gen_cost = T.nnet.binary_crossentropy(discriminator.act_out(generator.act_out(z)), y).mean()
    
    gen_updater = updates.Adam(lr=learning_rate_gen)
    gen_updates = gen_updater(generator.params, gen_cost)
    
    train_gen = theano.function(inputs = [z, y], outputs = [], updates = gen_updates)
    
    #########
    #########   LOAD DATA
    ######### 
    
    fp = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(fp)
    fp.close()

    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x,  test_y  = test_set

    data = numpy.concatenate((train_x, valid_x))
    
    num_batches = len(data) / batch_size
    
    #########
    #########   TRAIN
    ######### 

    for e in range(start, start+num_epochs):
    
        print '#     starting epoch', e, 'at', datetime.datetime.now()

        # train networks
        for i in range(num_batches):
        
            print '#       mini-batch', i+1, '/', num_batches, '\r',
            sys.stdout.flush()
        
            # randomly select a mini-batch of z-samples
            z_samples = numpy.random.uniform(-1, 1, size=(batch_size, n_z)).astype('float32')

            # train the generator on these
            y_vals = numpy.ones((batch_size, 1), dtype=theano.config.floatX)
            train_gen(z_samples, y_vals)
            
            # train the discriminator on these
            y_vals = numpy.zeros((batch_size, 1), dtype=theano.config.floatX)
            train_dis_fake(z_samples, y_vals)
                
            # train the discriminator on data
            y_vals = numpy.full((batch_size, 1), fill_value=smooth, dtype=theano.config.floatX)
            train_dis_data(data[i*batch_size : (i+1)*batch_size], y_vals )
    
        # image the results
        if True:
            num_images = 4
            for n in range(num_images):
                filen = 'out_' + str(e) + '_' + str(n) + '_.bmp'
                z_inp = numpy.random.uniform(-1, 1, size=(n_z)).astype('float32')
                vector = generator.act_out(z_inp).eval()
                square = numpy.reshape(vector, (28, 28))
                image = Image.fromarray(square, mode='1')
                image.save(filen)

    #########
    #########   PRESERVE NETWORK PARAMS 
    ######### 
                
    try:
        os.rename('param_.pkl', 'param_old_.pkl')            
    except:
        pass
    
    fp = open('param_.pkl', 'wb')
    cPickle.dump(generator.params, fp)
    cPickle.dump(discriminator.params, fp)
    fp.close()
    
############################################################################################################################################    

    
    
    
    
    
    
    
    
    
    
    