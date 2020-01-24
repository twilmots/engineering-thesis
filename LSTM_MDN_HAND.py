from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
import sys
import os

from MDNClass import MDN
from DataLoader import DataLoader
from gmm_sample import *

# Choose 1 or 2 for different ydata graphs
PLOT = 1

# Make everything float32 
d_type = tf.float32

# Batch size for training
train_batch_size = 50

# Number of steps (RNN rollout) for training
train_num_steps = 300

# Dimension of LSTM input/output
hidden_size = 3

# should we do dropout? (1.0 = nope)
train_keep_prob = 0.80

# number of training epochs
num_epochs = 100

# how often to print/plot
update_every = 10

# how often to save
save_every = 2

# initial weight scaling
init_scale = 0.1

# Number of things in our cascade
steps_in_cascade = 3

# Input dimension
input_dimension = 3

# Handle sequencing or not
handle_sequences_correctly = True

# do xy offsets or not
do_diff = True

# learning rate
learning_rate = 1e-4

# do we want gifs?! yes?
CREATE_GIFS = False

# do we want to generate handwriting 
GENERATE_HANDWRITING = True

# do we want to visualize with tensorboard
CREATE_TENSORBOARD = False

######################################################################
# Helper function for below

def get_xy_data(n):
	u = np.arange(n)*0.4 + np.random.random()*10
	if PLOT == 1:
		x = u
		y = 8.0*(np.abs((0.125*u - np.floor(0.125*u)) - 0.5)-0.25)
	else:
		x = u + 3.0*np.sin(u)
		y = -2.0*np.cos(u)
                
        x -= x.min()
        y -= y.min()
	return x, y

######################################################################
# Get training data -- the format returned is xi, yi, 0 except for new
# "strokes" which are xi, yi, 1 every time the "pen is lifted".
    
def get_data(data):
    cur_count = 0
    all_data = []
    all_sequence_info = []

    subsequence_index = 0

    for i in range(len(data)):
        sequence = data[i]
        length_sequence = len(sequence)
        
        # sequence info has two columns
        sequence_info = np.zeros((length_sequence,2),dtype = int)

        # first column is all ones expcet for 0 for the very first point in sequence
        sequence_info[0,0] = 0
        sequence_info[1:,0] = 1

        # second column just holds which subsequence we are on -- not used for training
        # used to visualize rows in PDFs

        sequence_info[:,1] = subsequence_index
        subsequence_index += 1

        all_sequence_info.append(sequence_info)
    
    all_sequence_info = np.vstack(tuple(all_sequence_info))
    all_data = np.vstack(tuple(data))

    return all_data, all_sequence_info

######################################################################
class Input(object):

    def __init__(self, posdata, seqinfo, config):

        batch_size = config.batch_size
        num_steps = config.num_steps
        self.posdata = posdata
        self.seqinfo = seqinfo

        # I think we need this name scope to make sure that each
        # condition (train/valid/test) has its own unique producer?
        with tf.name_scope('producer', [posdata, batch_size, num_steps]):

            # Convert original raw data to tensor
            raw_data = tf.convert_to_tensor(posdata, name='raw_data', dtype=d_type)

            # Convert sequence continuations to tensor - just want for column
            raw_seq = tf.convert_to_tensor(seqinfo[:,0], name = 'sef_info', dtype=d_type)

            # These will be tensorflow variables
            data_len = tf.size(raw_data)//3
            batch_len = data_len // batch_size
            epoch_size = (batch_len - 1) // num_steps

            # Prevent computation if epoch_size not positive
            assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")

            with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

            # Truncate our raw_data and reshape it into batches
            # This is just saying grab as much of it as we can to make a clean reshaping
            data = tf.reshape(raw_data[:batch_size*batch_len, :],
                              [batch_size, batch_len, 3])

            seq = tf.reshape(raw_seq[:batch_size*batch_len],
                             [batch_size, batch_len])

            # i is a loop variable that indexes which batch we are on
            # within an epoch
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

            # each slice consists of num_steps*batch_size examples
            x = tf.slice(data, [0, i*num_steps, 0], [batch_size, num_steps, 3])
            y = tf.slice(data, [0, i*num_steps+1, 0], [batch_size, num_steps, 3])

            preserve_state = tf.slice(seq, [0, i*num_steps], [batch_size, num_steps])
            err_weight = tf.slice(seq, [0, i*num_steps+1], [batch_size, num_steps])
        # Assign member variables
        self.x = x
        self.y = y
        self.epoch_size = ((len(posdata) // batch_size)-1) // num_steps

        self.preserve_state = preserve_state
        self.err_weight = tf.reshape(err_weight, [batch_size, num_steps, 1])


######################################################################
# Class of Cascading LSTMs 
class LSTMCascade(object):

    def __init__(self, config, model_input, is_train, is_sample=False, external_targets=None):

        # Stash some variables from config
        hidden_size = config.hidden_size
        batch_size = config.batch_size
        num_steps = config.num_steps
        keep_prob = config.keep_prob

        # Scale factor so we can vary dataset size and see "average" loss
        # Do this in case we're just looking at a single point and we're querying
        self.loss_scale = batch_size * num_steps * model_input.epoch_size

        # Stash input
        self.model_input = model_input

        # we don't need to reshape the data!
        if is_sample:
            self.lstm_input = tf.placeholder(tf.float32, shape = [None,1,3])
            model_input.y = tf.zeros(shape=[1,1,3])
        else:
            self.lstm_input = model_input.x

        # this is going to be the final dimension 
        # this is always even
        final_high_dimension = input_dimension * steps_in_cascade * (steps_in_cascade+1) // 2

        # note: input dimension is equivalent to the hidden size of the LSTM cell
        hidden_size = input_dimension

        # this will hold all of our cells
        lstm_stack = []

        # this will hold all of our states as it goes
        self.state_stack = []

        # this will hold the initial states
        init_state_stack = []

        # This will reduce our final outputs to the appropriate lower dimension
        # Make weights to go from LSTM output size to 2D output
        w_output_to_y = tf.get_variable('weights_output_to_y', [final_high_dimension, 2],
                                        dtype=d_type)

        # we need to # LSTMS = # steps in cascade
        for i in range(steps_in_cascade):

            # Make an LSTM cell
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                hidden_size * (i+1), forget_bias=0.0,
                state_is_tuple=True)

            # Do dropout if needed
            if is_train and keep_prob < 1.0:
                print('doing dropout with prob {}'.format(config.keep_prob))
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=keep_prob)

            initial_state = lstm_cell.zero_state(batch_size, d_type)

            lstm_stack.append(lstm_cell)
            init_state_stack.append(initial_state)
            self.state_stack.append(initial_state)

        # cache our initial states
        self.initial_state = init_state_stack

        # Need an empty total output list of ys
        outputs = []

        # we need this variable scope to prevent us from creating multiple
        # independent weight/bias vectors for LSTM cell
        with tf.variable_scope('RNN'):

            # For each time step
            for time_step in range(num_steps):

                # This is y_i for a single time step
                time_step_output = []

                # Prevent creating indep weights for LSTM
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                # model_input.preserve_state is a vector of 0's or 1's corresponding
                # to 0 means reset LSTM, 1 means don't
                preserve_step = tf.reshape(model_input.preserve_state[:, time_step],
                                           [config.batch_size,1])

                for i in range(steps_in_cascade):

                    with tf.variable_scope("RNN"+str(i)):
                        # Run the lstm cell using the current timestep of
                        # input and the previous state to get the output and the new state
                        curr_lstm_cell = lstm_stack[i]
                        curr_state = self.state_stack[i]


                        # state.c and state.h are both shape (batch_size, hidden_size)
                        # when I multiply by (batch_size, 1) it broadcasts
                        curr_stateTuple = type(curr_state)
                        possible_state = curr_stateTuple(c = curr_state.c*preserve_step,
                                                         h = curr_state.h*preserve_step)

                        # Need a special base case for the first lstm input 
                        if i == 0:
                            cell_input = self.lstm_input[:, time_step, :]
                        else:
                            # All of these variables will be defined because of our base case
                            cell_input = tf.concat(concat_dim = 1, values = [self.lstm_input[:, time_step, :], cell_output])

                        (cell_output, curr_state) = curr_lstm_cell(cell_input,
                                             possible_state)

                        # Update our state list
                        self.state_stack[i] = curr_state

                        # Update the output for the single cell
                        time_step_output.append(cell_output)

                # For every timestep, we need a valid y output that should be of N*L*(L+1)/2 
                concated_time_steps = tf.concat(concat_dim = 1 , values = time_step_output)
                outputs.append(concated_time_steps)

        # we need to bookmark the final state to preserve continuity
        # across batches when we run an epoch (see below)
        # note, this is a list
        self.final_state = self.state_stack

        # concatenate all the outputs together into a big rank-2
        # matrix where each row is of dimension hidden_size
        # not sure what this concatenate is doing
        lstm_output_rank2 = tf.reshape(tf.concat(1, outputs), [-1, final_high_dimension])

        if external_targets is None:
            # reshape original targets down to rank-2 tensor
            targets_rank2 = tf.reshape(model_input.y, [batch_size*num_steps, 3])
        else:
            targets_rank2 = tf.reshape(external_targets, [-1, 3])

        with tf.variable_scope('MDN'):
            ourMDN = MDN(lstm_output_rank2, targets_rank2, final_high_dimension, is_train)
            self.pis, self.corr, self.mu, self.sigma, self.eos = ourMDN.return_params()

        # The loss is now calculated from our MDN
        MDNloss, log_loss = ourMDN.compute_loss()
        self.log_loss = log_loss

        if external_targets is None:
            log_loss = tf.reshape(log_loss, [batch_size, num_steps,1])

            loss = log_loss * model_input.err_weight
            self.loss_by_err_wt = loss
            # err_weight = tf.slice(seq, [0, i*num_steps+1], [batch_size, num_steps])
            # What we now care about is the mixture probabilities from our MDN
        else:
            loss = MDNloss

        with tf.variable_scope('MDN'):
            self.mixture_prob = ourMDN.return_mixture_prob()
            self.ncomponents = ourMDN.NCOMPONENTS

        # loss is calculated in our MDN
        self.loss = tf.reduce_sum(loss)
        self.loss_before_max = self.loss
        self.err_wt_reduce_sum = tf.reduce_sum(model_input.err_weight)
        self.loss /= tf.maximum(tf.reduce_sum(model_input.err_weight),1)
        self.after_max_division = self.loss
        # generate a train_op if we need to
        if is_train:
            self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        else:
            self.train_op = None


    def run_epoch(self, session, return_predictions=False, query=False):
        # we always fetch loss because we will return it, we also
        # always fetch the final state because we need to pass it
        # along to the next batch in the loop below.
        # final state is now a list!! Update!! of three state tensors
        fetches = {
            'loss': self.loss,
            'final_state': self.final_state,
            'log loss' : self.log_loss,
	       'loss before max': self.loss_before_max,
            'err wt reduce sum': self.err_wt_reduce_sum,
	       'after tf.max div': self.after_max_division,
        }

        # we need to run the training op if we are doing training
        if self.train_op is not None:
            fetches['train_op'] = self.train_op

        # we need to fetch the network outputs if we are doing predictions
        if return_predictions:
            fetches['p'] = self.mixture_prob

        # run the initial state to feed the LSTM - this should just be
        # zeros
        state = session.run(self.initial_state)

        # we will sum up the total loss
        total_loss = 0.0

        all_outputs = []

        ##################################################
        # for each batch:
        
        for step in range(self.model_input.epoch_size):

            for level in range(len(state)):
                # the input producer will take care of feeding in x/y,
                # but we need to feed in the LSTM state
                c, h = self.initial_state[level]
                feed_dict = { c: state[level].c, h: state[level].h }

                # run the computation graph?
                vals = session.run(fetches, feed_dict)
                #print(vals)

                # get the final LSTM state for the next iteration
                state = vals['final_state']

            # stash output if necessary
            if return_predictions:
                all_outputs.append(vals['p'])

            # update total loss
            total_loss += vals['loss']

        # do average
        total_loss /= self.loss_scale

        # return one or two things
        if not return_predictions:
            return total_loss
        elif query:
            return total_loss, vals['p']
        else:
            return total_loss, np.vstack(all_outputs)

    def query(self, input_data, y, curr_state_list):

        return self.mixture_prob

    def sample(self, session):
        # this is a list of three states
        prev_state = session.run(self.initial_state)

        fetches = [self.pis, self.corr, self.mu, self.sigma, self.eos, self.final_state]

        c, h = self.initial_state[0]

        feed_dict = {
            c: prev_state[0].c,
            h: prev_state[0].h,
        }

        pis, corr, mu, sigma, eos, next_state = session.run(fetches, feed_dict)

        sample = gmm_sample(mu.reshape(-1,self.ncomponents,2), sigma.reshape(-1,self.ncomponents,2), corr, pis, eos)

        return sample
                
def sample(session, generate_config, initializer, duration=200):

    prev_x = np.zeros((2,1,3), dtype=np.float32)

    prev_x[0,0,2] = 1

    writing = np.zeros((duration,3), dtype = np.float32)

    for i in range(duration):
        print("At iteration {}".format(i))
        generate_data, generate_seq = get_data(prev_x)
        if i < 5:
            print(generate_data)

        with tf.name_scope('generate'+str(i)):
            generate_input = Input(generate_data, generate_seq, generate_config)

            with tf.variable_scope('model', reuse=True, initializer=initializer):
                generate_model = LSTMCascade(generate_config, generate_input, is_sample=False, is_train=False)

        tf.train.start_queue_runners(session)
        
        sample_pt = generate_model.sample(session)
        print('sample pt shape: {}'.format(sample_pt.shape))
        print('prev_x shape: {}'.format(prev_x.shape))

        prev_x = np.vstack((prev_x, sample_pt.reshape(-1,1,3)))

        writing[i,:] = sample_pt

    return writing

######################################################################
# plot input vs predictions

def integrate(xyoffs, seq):

    # split up into subsequences
    n = xyoffs.shape[0]
    
    start_indices = np.nonzero(seq[:,0] == 0)[0]

    all_outputs = []

    for i, start_idx in enumerate(start_indices):
        if i + 1 < len(start_indices):
            end_idx = start_indices[i+1]
        else:
            end_idx = n
        xyslice = xyoffs[start_idx:end_idx]
        all_outputs.append(np.cumsum(xyslice, axis=0))

    return np.vstack(tuple(all_outputs))

def make_plot(epoch, loss, data, seq, pred):

    titlestr = '{} test set loss = {:.2f}'.format(epoch, loss)
    print(titlestr)

    y = seq[:,1] * 6

    if do_diff:
        data = integrate(data, seq)
        pred = integrate(pred, seq)

    plt.clf()
    plt.plot(data[:,0], data[:,1]+y, 'b.')
    plt.plot(pred[:,0], pred[:,1]+y, 'r.')
    plt.axis('equal')
    plt.title(titlestr)
    plt.savefig('test_data_pred_lstm_3.pdf')

def make_handwriting_plot(generated_data, generated_seq):
    titlestr = 'Generated Handwriting'
    if do_diff:
        data = integrate(generated_data, generated_seq)
    plt.clf()
    plt.plot(data[:,0], data[:,1], 'r.', markersize=3)
    plt.axis('equal')
    plt.title(titlestr)
    plt.savefig('GeneratedHW.pdf')

def make_heat_plot(epoch, loss, query_data, seq, xrng, yrng, xg, pred, i):
    p = pred.reshape(xg.shape)
    titlestr = '{} query set loss = {:.2f}'.format(epoch,loss)

    y = seq[:,1] * 6
    query_data = integrate(query_data, seq)
    last_point = query_data[-1]
    plt.clf()
    ax = plt.gca()
    xdata = xrng+last_point[0]
    ydata = -(yrng+last_point[1])
    plt.pcolormesh(xdata, ydata, p, cmap='jet')
    plt.plot(query_data[:,0], -query_data[:,1], 'wo', alpha = 0.90, markersize=3)
    plt.axis('equal')
    plt.axis([xdata.min(), xdata.max(), ydata.min(), ydata.max()])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # plt.show()
    plt.title(titlestr)
    plt.savefig('Gifs/LSTMHeatMap' + str(i) + '.pdf', bbox_inches='tight', pad_inches = 0)

def make_heat_plot_no_integrate(epoch, loss, query_data, xrng, yrng, xg, pred, i):
    p = pred.reshape(xg.shape)
    titlestr = '{} query set loss = {:.2f}'.format(epoch,loss)

    last_point = query_data[-1]
    plt.clf()
    ax = plt.gca()
    xdata = xrng+last_point[0]
    ydata = -(yrng+last_point[1])
    plt.pcolormesh(xdata, ydata, p, cmap='jet')
    plt.plot(query_data[:,0], -query_data[:,1], 'wo', alpha = 0.90, markersize=5)
    plt.axis('equal')
    plt.axis([xdata.min(), xdata.max(), ydata.min(), ydata.max()])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # plt.show()
    plt.title(titlestr)
    plt.savefig('NewGifs/LSTMHeatMap' + str(i) + '.pdf', bbox_inches='tight')

######################################################################
# main function
    
def main():

    # configs are just named tuples
    Config = namedtuple('Config', 'batch_size, num_steps, hidden_size, keep_prob')

    # generate training and test configurations
    train_config = Config(batch_size=train_batch_size,
                          num_steps=train_num_steps,
                          hidden_size=hidden_size,
                          keep_prob=train_keep_prob)
    
    test_config = Config(batch_size=1,
                         num_steps=1,
                         hidden_size=hidden_size,
                         keep_prob=1)

    query_config = Config(batch_size= 1,
                          num_steps = 1,
                          hidden_size = hidden_size,
                          keep_prob = 1)

    generate_config = Config(batch_size = 1,
                           num_steps = 1,
                           hidden_size = hidden_size,
                           keep_prob = 1)

    # range to initialize all weights to
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    # Import our handwriting data
    data = DataLoader()

    our_train_data = data.data[0:50]
    our_valid_data = data.valid_data[0:50]
    our_query_data = data.valid_data[225:227]

    # generate our train data
    train_data, train_seq = get_data(our_train_data)

    # get our validation data
    valid_data, valid_seq = get_data(our_valid_data)

    # get the query data
    query_data, query_seq = get_data(our_query_data)
    query_data, query_seq = query_data[0:145, :], query_seq[0:145,:]

    # Let's get our mesh grid for visualization
    int_query_data = integrate(query_data, query_seq)
    # int_query_y = query_seq[:,1] * 6
    # itq = -int_query_data[:,1] + int_query_y

    last_point = int_query_data[-1]
    xmin, xmax = (int_query_data[:,0]-last_point[0]).min()-10, (int_query_data[:,0]-last_point[0]).max()+10
    ymin, ymax = ((int_query_data[:,1]-last_point[1]).min()-10), ((int_query_data[:,1]-last_point[1]).max()+10)
    print('xmin: {} xmax: {} \n ymin: {} ymax: {}'.format(xmin,xmax,ymin,ymax))

    xrng = np.linspace(xmin, xmax, 200, True)
    yrng = np.linspace(ymin, ymax, 200, True)

    xg, yg = np.meshgrid(xrng, yrng)

    xreshape, yreshape = xg.reshape(-1,1), yg.reshape(-1,1)
    third_col = np.ones(xreshape.shape)

    mesh_target = np.hstack([xreshape, yreshape, third_col])
    mesh_target = mesh_target.reshape(-1, 1, 3).astype('float32')

    # generate input producers and models -- again, not 100% sure why
    # we do the name_scope here...
    with tf.name_scope('train'):
        train_input = Input(train_data, train_seq, train_config)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            train_model = LSTMCascade(train_config, train_input, is_train=True)

    # with tf.name_scope('valid'):
    #     valid_input = Input(valid_data, valid_seq, train_config)
    #     with tf.variable_scope('model', reuse=True, initializer=initializer):
    #         valid_model = LSTMCascade(train_config, train_input, is_train=False)
            
    # with tf.name_scope('test'):
    #     test_input = Input(test_data, test_config)
    #     with tf.variable_scope('model', reuse=True, initializer=initializer):
    #         test_model = LSTMCascade(test_config, test_input, is_train=False)

    with tf.name_scope('query'):
        query_input = Input(query_data, query_seq, query_config)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            query_model = LSTMCascade(query_config, query_input, is_train=False, external_targets=mesh_target)
    
    prev_x = np.zeros((2,1,3), dtype = np.float32)
    generate_data, generate_seq = get_data(prev_x)
    print('generate seq: {}'.format(generate_seq))

    with tf.name_scope('generate'):
        generate_input = Input(generate_data, generate_seq, generate_config)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            generate_model = LSTMCascade(generate_config, generate_input, is_sample=True, is_train=False)
    
    if CREATE_GIFS:
        query_models = []
        for i in range(2,len(query_data)):
            with tf.name_scope('gif_query'+str(i)):
                seg_query_data = query_data[0:i,:]
                seg_query_seq = query_seq[0:i,:]

                int_seg_query_data = integrate(seg_query_data, seg_query_seq)

                last_point = int_seg_query_data[-1]
                xmin, xmax = (int_seg_query_data[:,0]-last_point[0]).min()-10, (int_seg_query_data[:,0]-last_point[0]).max()+10
                ymin, ymax = ((int_seg_query_data[:,1]-last_point[1]).min()-10), ((int_seg_query_data[:,1]-last_point[1]).max()+10)

                xrng = np.linspace(xmin, xmax, 200, True)
                yrng = np.linspace(ymin, ymax, 200, True)

                xg, yg = np.meshgrid(xrng, yrng)

                xreshape, yreshape = xg.reshape(-1,1), yg.reshape(-1,1)
                third_col = np.ones(xreshape.shape)

                mesh_target = np.hstack([xreshape, yreshape, third_col])
                mesh_target = mesh_target.reshape(-1, 1, 3).astype('float32')

                query_input = Input(seg_query_data, seg_query_seq, query_config)
                with tf.variable_scope('model', reuse=True, initializer=initializer):
                    query_models.append(LSTMCascade(query_config, query_input, is_train=False, external_targets=mesh_target))

    # print out all trainable variables:
    tvars = tf.trainable_variables()
    print('trainable variables:')
    print('\n'.join(['  - ' + tvar.name for tvar in tvars]))

    # create a session
    session = tf.Session()

    # # let's save our computation graph IF we don't already have a parameter
    saver = tf.train.Saver()
    
    # need to explicitly start the queue runners so the index variable
    # doesn't hang. (not sure how PTB did this - I think the
    # Supervisor takes care of it)
    tf.train.start_queue_runners(session)

    if len(sys.argv) > 1:

        saver.restore(session, sys.argv[1])

        print('Did a restore. Here are all the variables:')
        
        tvars = tf.global_variables()
        print('\n'.join(['  - ' + tvar.name for tvar in tvars]))

        # l, pred = query_model.run_epoch(session, return_predictions=True, query=True)
        # make_heat_plot('Model {}'.format(0), l, query_data, query_seq, xrng, yrng, xg, pred, 1000)

        if CREATE_GIFS:
            for idx, model in enumerate(query_models):
                int_query_data = integrate(model.model_input.posdata, model.model_input.seqinfo)
                last_point = int_query_data[-1]
                xmin, xmax = (int_query_data[:,0]-last_point[0]).min()-10, (int_query_data[:,0]-last_point[0]).max()+10
                ymin, ymax = ((int_query_data[:,1]-last_point[1]).min()-10), ((int_query_data[:,1]-last_point[1]).max()+10)

                xrng = np.linspace(xmin, xmax, 200, True)
                yrng = np.linspace(ymin, ymax, 200, True)
                l, pred = model.run_epoch(session,return_predictions=True, query=True)
                make_heat_plot_no_integrate('Model {}'.format(idx), l, int_query_data, xrng, yrng, xg, pred, idx)

        if GENERATE_HANDWRITING:
            # not sure what model we should pass in
            writing = sample(session, generate_config, initializer)
            seq = np.ones(shape = (writing.shape[0], 1))
            seq[0,0] = 0
            make_handwriting_plot(writing, seq)
            print('Handwriting generated.')

        if CREATE_TENSORBOARD:
            writer = tf.summary.FileWriter("tensorboard_output", session.graph)
            writer.close()

    else:

        # initialize all the variables
        session.run(tf.global_variables_initializer())

        # for each epoch
        for epoch in range(num_epochs):

            # run the epoch & get training loss
            l = train_model.run_epoch(session)
            print('training loss at epoch {}    is {}'.format(epoch, l))
            if epoch % save_every == 0:

                print('Saving model..... ')

                if not os.path.isdir('models'):
                    os.mkdir('models')

                written_path = saver.save(session, 'models/rnn_demo',
                          global_step=epoch)
                print('saved model to {}'.format(written_path))

            # see if we should do a printed/graphical update
            if epoch % update_every == 0:

                print()

                l = valid_model.run_epoch(session)
                print('validation loss at epoch {} is {:.2f}'.format(epoch, l))

                l, pred = query_model.run_epoch(session, return_predictions=True, query=True)
                # make_heat_plot('epoch {}'.format(epoch), l, query_data, query_seq, xrng, yrng, xg, pred, epoch)

                # if not os.path.isdir('models'):
                #     os.mkdir('models')

                written_path = saver.save(session, 'models/backup/rnn_demo',
                          global_step=epoch)
                # print('saved model to {}'.format(written_path))

                print()

        written_path = saver.save(session, 'models/rnn_demo', global_step=num_epochs)
        print('saved final model to {}'.format(written_path))
        # do final update
        l, pred = query_model.run_epoch(session, return_predictions=True, query=True)
        make_heat_plot('final', l, query_data, seq, xrng, yrng, xg, pred, 1000)
    

if __name__ == '__main__':

    main()
