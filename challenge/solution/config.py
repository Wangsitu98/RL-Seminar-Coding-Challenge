#!/usr/bin/python
import random
import tensorflow as tf
class Config:
	seed = random.randint(1, 1000)
	num_workers = 1
	
	lr = 0.01
	state_shape = [None , 10, 10, 1]
	action_size = 100
	state_size = 2
	batchsize = 32
	bias_initializer = tf.random_normal_initializer(seed = seed)
	activation = tf.nn.elu
	
	alpha = 0.5
	alpha_decay = 0.999
	gamma = 0.999
	random_action_taken_at_begining = 0
	drop_rate = 0.
	opt = tf.train.AdamOptimizer(learning_rate = lr)
	
	model_path = "./model"
	load_model = False
	max_episode_length = 300
	