#!/usr/bin/python
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.signal
from config import Config as cfg

def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer
	
def dropout(inputs):
	return tf.nn.dropout(inputs, keep_prob = 1 - cfg.drop_rate)


	

class A3C_Network():
	def __init__(self,s_size, a_size, scope, trainer):
		with tf.variable_scope(scope):
			#Input and visual encoding layers
			self.inputs = tf.placeholder(shape=cfg.state_shape, dtype=tf.float32)
			
			conv = tf.layers.conv2d(self.inputs, filters = 1, kernel_size = 3, padding = 'same', activation=tf.nn.leaky_relu)
			hidden = tf.layers.flatten(conv)
			
			self.policy = tf.nn.softmax(hidden)
			
			v1 = tf.nn.leaky_relu(tf.layers.dense(hidden, 32))
			self.value = tf.layers.dense(hidden, 1)
			
			#Only the worker network need ops for loss functions and gradient updating.
			if scope != 'global':
				self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
				self.actions_onehot = tf.one_hot(self.actions, cfg.action_size ,dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

				#Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
				self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
				self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
				self.loss = 1.0 * self.value_loss + 1.0 * self.policy_loss - 0.1 * self.entropy

				#Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss, local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
				
				#Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
				
