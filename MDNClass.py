import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Prediction output dimension
output_dimension = 2

# Make everything float32 
d_type = tf.float32


class MDN:

	def __init__(self, input_data_from_lstms, targets, final_dimension_from_lstm, is_training):
		# The input is now longer raw data 
		# The input is now the output of our LSTMs
		# We will use this as a module in our model
		self.NHIDDEN = 30 
		self.NCOMPONENTS = 15
		self.STEP = final_dimension_from_lstm
		self.ndim = output_dimension
		self.data = input_data_from_lstms
		self.iterations = 8000

		# Changed
		# self.x = x = tf.placeholder(tf.float32, shape = [None, self.STEP*self.ndim]) 
		self.x = x = input_data_from_lstms

		# NOTE: not doing this for the combination of LSTM + MDN
		# Set up the hidden layer
		# # wh = tf.Variable(tf.random_normal([self.STEP * self.ndim, self.NHIDDEN]), name='wh')
		# wh = tf.Variable(tf.random_normal([self.STEP, self.NHIDDEN], dtype = d_type), name='wh')
		# bh = tf.Variable(tf.random_normal([self.NHIDDEN], dtype = d_type), name='bh')

		# # Add dropout
		# # wh = tf.nn.dropout(wh, keep_prob = 1.0)

		# # Train
		# temp_h = tf.add(tf.matmul(x, wh), bh)
		# self.h = h = tf.nn.tanh(temp_h) # Shape (?, hidden) = (?, 30)

		''' Here, we are building the MDN'''
		# Setting up the Pi's (they are one dimensional)
		w_pi = tf.get_variable(name = 'w_pi', initializer = tf.random_normal([final_dimension_from_lstm, self.NCOMPONENTS], dtype = d_type))
		b_pi = tf.get_variable(name = 'b_pi', initializer = tf.random_normal([self.NCOMPONENTS], dtype = d_type))
		p = tf.add(tf.matmul(x, w_pi),b_pi)
		pis = tf.exp(p)
		pis = pis/tf.reshape(tf.reduce_sum(pis, axis=1), [-1, 1]) # Shape (?, number of components) = (?, 3)	
		self.pis = pis

		# Setting up the correlations
		w_corr = tf.get_variable(name = 'w_corr', initializer = tf.random_normal([final_dimension_from_lstm, self.NCOMPONENTS], stddev = 0.2, dtype = d_type))
		b_corr = tf.get_variable(name = 'b_corr', initializer = tf.random_normal([self.NCOMPONENTS], stddev = 0.2, dtype = d_type))
		precorr = tf.add(tf.matmul(x,w_corr), b_corr)
		corr = tf.tanh(precorr) # Shape (?, number of components) = (?, 3)
		self.corr = corr

		# Setting up the means
		# Changed!!: 
		w_mu = tf.get_variable(name = 'w_mu', initializer = tf.random_normal([final_dimension_from_lstm, self.NCOMPONENTS*self.ndim], dtype = d_type))
		b_mu = tf.get_variable(name = 'b_mu', initializer = tf.random_normal([self.NCOMPONENTS*self.ndim], dtype = d_type))

		# no activation function for the mus 
		mu = tf.add(tf.matmul(x,w_mu),b_mu) # Shape (?, number of comp * number of dimensions) = (?, 6)
		self.mu = mu

		# Setting up sigma and variance
		# Changed!!:
		w_sigma = tf.get_variable(name = 'w_sigma', initializer = tf.random_normal([final_dimension_from_lstm, self.NCOMPONENTS*self.ndim], dtype = d_type))
		b_sigma = tf.get_variable(name = 'b_sigma', initializer = tf.random_normal([self.NCOMPONENTS*self.ndim], dtype = d_type))
		sigma = tf.add(tf.matmul(x,w_sigma),b_sigma)
		sigma = tf.exp(sigma) # Shape (?, number of comp * number of dimensions) = (?, 6)
		self.sigma = sigma

		# end of stroke parameter
		eos = tf.get_variable(name = 'eos', initializer = tf.random_normal([1], dtype = d_type))
		self.eos = eos = tf.divide(1.0, tf.add(1.0,tf.exp(eos))) 


		''' Here we are going to build the mixture probabilites'''
		sum_of_pis = tf.reduce_sum(pis)

		# Target values
		xy_targets = tf.slice(targets, [0, 0], [-1, 2])
		eos_targets = tf.slice(targets, [0, 2], [-1, 1])
		self.actual = actual = tf.reshape(xy_targets, [-1, 1, self.ndim])

		# Need to first do some dimensionality manipulation
		# specifically for mu and sigma
		# need to reshape with the 3 first, so that our x1 and x2 points get multiplied correctly
		mu = tf.reshape(mu, [-1, self.NCOMPONENTS, output_dimension])
		sigma = tf.reshape(sigma, [-1, self.NCOMPONENTS, output_dimension])

		# The output we feed into this function should be that from the hidden layers
		var = tf.mul(sigma,sigma) # Shape (?, 3, 2)

		# Compress along the number of gaussians so that we still have the shape: 1 x 3 )
		s1s2 = tf.reduce_prod(sigma, axis=2) # Shape: (?,3) 
		# s1s2 = tf.reshape(s1s2,[-1]) # Shape: (3,) # CHECK SHAPE HERE

		# deviation from mean is [?, n_components]
		# so x is going to be like 150 x 2 and then mus is going to be 3 x 2
		# actual - (?, 2, 1) 
		# mu - (?, 6)
		dev = tf.sub(actual,mu) # Shape: (?, 3, 2)
		# Shape explanation: we have a bunch of sequences and we have x and y points
		#					 for three different components

		# Have to build Z equations first (eqn 25)
		# Broadcasting works - ? x 3 x 2
		z12_before = tf.div(tf.mul(dev,dev),var)
		# shape above: (?, 2, 3)
		z12 = tf.reduce_sum(z12_before, axis=2)
		# shape above: (?, 3)

		# dev is ? x 3 x 2. the six comes from numberofcomp * number of dimensions 
		# after reduce prod it's ? x 3 which is good. 
		# six is good because we have three components and each has an x and a y!!
		reduce_dev = tf.reduce_prod(dev, axis=2) # Shape: (?, 3)

		# z3 is going to be (?, 3)
		# this math also works out and makes sense
		z3 = tf.div(reduce_dev*corr*2, s1s2)
		Z = z12 - z3

		# Building Normal Distribution (eqn 24)
		normalizer = (2.0 * np.pi * s1s2) * tf.sqrt(1.0-tf.mul(corr,corr))

		expon_part = tf.exp(tf.div(-Z, 2.0 * (1.0 - tf.mul(corr,corr))))

		N = tf.div(expon_part, normalizer) # Shape: (?, 3)

		# Building conditional probability (eqn 23)
		# overall mixture probabilities has shape [?]
		N_by_pis = tf.mul(N,pis)
		self.N_by_pis = N_by_pis
		
		eos_operator = tf.mul(eos,eos_targets) + tf.mul(1.0-eos, 1.0-eos_targets)
		self.eos_operator = eos_operator

		N_by_pis_by_eos = tf.mul(N_by_pis, eos_operator)

		self.mixture_prob = mixture_prob = tf.reduce_sum(N_by_pis_by_eos, axis=1)

	def compute_loss(self):
		mixture_prob = self.mixture_prob

		negative_log_odds = -tf.log(tf.maximum(mixture_prob,1e-20)) 
		loss = tf.reduce_sum(negative_log_odds)

		return loss, negative_log_odds

	def return_mixture_prob(self):
		return self.mixture_prob

	def return_params(self):
		return self.pis, self.corr, self.mu, self.sigma, self.eos

