#!/usr/bin/python
import threading
import os
import gym
import coding_challenge
import numpy as np

from config import Config as cfg
from A3C_Net import A3C_Network
from Worker import Worker
import tensorflow as tf
from time import sleep


a_size = cfg.action_size
s_size = cfg.state_shape

if not os.path.exists(cfg.model_path):
	os.makedirs(cfg.model_path)
	
if not os.path.exists('./frames'):
	os.makedirs('./frames')

with tf.device("/cpu:0"): 
	global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
	opt = cfg.opt
	master_network = A3C_Network(s_size, a_size, 'global', None)
	num_workers = cfg.num_workers
	workers = []

	for i in range(cfg.num_workers):
		workers.append(Worker(gym.make('Battleship-v0'), i, s_size, a_size, opt, cfg.model_path, global_episodes))
	saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	if cfg.load_model == True:
		print ('Loading Model...')
		saver.restore(sess, cfg.model_path+"/checkpoint.ckpt")
	else:
		sess.run(tf.global_variables_initializer())
		
		
	worker_threads = []
	for worker in workers:
		worker_work = lambda: worker.work(cfg.max_episode_length, cfg.gamma, sess, coord, saver)
		t = threading.Thread(target=(worker_work))
		t.start()
		sleep(0.5)
		worker_threads.append(t)
	coord.join(worker_threads)