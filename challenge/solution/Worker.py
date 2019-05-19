#!/usr/bin/python
from config import Config as cfg
import tensorflow as tf
import numpy as np
from A3C_Net import A3C_Network
from random import choice
import scipy.signal
import random
import math


def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder


def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
	
def randomize_state(env):
	done = True
	ini_r = 0
	while done:
		done = False
		obs = env.reset()
		for i in range(cfg.random_action_taken_at_begining):
			action = env.action_space.sample()
			obs, reward, done, info = env.step(action)
			ini_r += reward
	
	assert(done == False)
	return env, obs, ini_r

def map_action(act):
	x, y = divmod(act, 10)
	'''
	print((x, y))
	print((x/10, y/10))
	'''
	return x/10, y/10
		
	

class Worker():
	def __init__(self, game, name,s_size, a_size, trainer, model_path, global_episodes):
		self.name = "worker_" + str(name)
		self.number = name        
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
		self.local_AC = A3C_Network(s_size,a_size,self.name,trainer)
		self.update_local_ops = update_target_graph('global',self.name)        	
		self.actions = np.identity(9,dtype=bool).tolist()
		self.env = game
		
	def train(self, rollout, sess, gamma, bootstrap_value):
		rollout = np.array(rollout)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]
		
		# Here we take the rewards and values from the rollout, and use them to 
		# generate the advantage and discounted returns. 
		# The advantage function uses "Generalized Advantage Estimation"
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs:np.vstack(observations).reshape(-1 , 10, 10, 1),
			self.local_AC.actions:actions,
			self.local_AC.advantages:advantages}
		v_l, p_l, e_l, g_n, v_n,_ = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.apply_grads],
			feed_dict=feed_dict)
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
		
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print ("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():                 
			while not coord.should_stop():
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				d = False
				
				self.env, s, ini_r= randomize_state(self.env)
				episode_reward += ini_r
				
				episode_frames.append(s)
				while d == False:
					#Take an action using probabilities from policy network output.
					act_dist, v = sess.run([self.local_AC.policy,self.local_AC.value], 
						feed_dict={self.local_AC.inputs:[s]})
					act_dist = act_dist[0]
					act = np.unravel_index(np.argmax(act_dist),  act_dist.shape)[0]
					s_new, r, d, info = self.env.step(map_action(act))
					print(info['game_message'])
					if d == False:
						s1 = s_new
						episode_frames.append(s1)
						s1 = s1
					else:
						s1 = s
						
					episode_buffer.append([s, act, r, s1, d, v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s = s1                    
					total_steps += 1
					episode_step_count += 1
					
					# If the episode hasn't ended, but the experience buffer is full, then we
					# make an update step using that experience rollout.
					if len(episode_buffer) == cfg.batchsize and d != True and episode_step_count != max_episode_length - 1:
						# Since we don't know what the true final return is, we "bootstrap" from our current
						# value estimation.
						v1 = sess.run(self.local_AC.value, 
							feed_dict={self.local_AC.inputs:[s]})[0,0]
						v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer, sess, gamma, v1)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if d == True or episode_step_count >= max_episode_length - 1:
						break
											
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				
				# Update the network using the episode buffer at the end of the episode.
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
								
					
				# Periodically save gifs of episodes, model parameters, and summary statistics.
				if episode_count % 5 == 0 and episode_count != 0:
					if episode_count % 5 == 0 and self.name == 'worker_0':
						saver.save(sess, cfg.model_path+"/checkpoint.ckpt")
						print ("Saved Model")

					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_length = np.mean(self.episode_lengths[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				if self.name == 'worker_0':
					sess.run(self.increment)
					
				episode_count += 1
				
				print("the reward for thread " + self.name + " in episode " + str(episode_count) + " is *"+ str(episode_reward) + "* in " +str(episode_step_count) + " time")

