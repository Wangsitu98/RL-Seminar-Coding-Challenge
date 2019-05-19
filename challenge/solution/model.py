#!/usr/bin/python
import tensorflow as tf
class Model():
	def __init__(self, sess, shape=[-1, 10, 10, 1]):
		self.sess = sess
		
		self.x = tf.placeholder(shape=shape)
		x = tf.layers.flatten(self.x)
		l1 = tf.nn.relu(tf.layers.dense(x, 100))
		l2 = tf.nn.relu(tf.layers.dense(l1, 100))
		self.action = tf.nn.softmax(l1, dim=2)
		
		