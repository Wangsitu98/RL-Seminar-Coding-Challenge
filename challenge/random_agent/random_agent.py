import gym
import coding_challenge
import numpy as np

"""
Creates an environment of the Battleship game and plays one episode in it by randomly choosing the actions.
The actions handed to the Battleship environment should be 2 dimensional (x and y position of the bomb you want to
drop) and should be scaled to the range [0,1]. The actions are internally discretized to the 10 by 10 playing field.
"""
env = gym.make('Battleship-v0')
state = env.reset()
terminal = False
rewards = 0
while not terminal:
    action = np.random.rand(2)
    state, reward, terminal, info = env.step(action)
    rewards = rewards + reward
    print(info['game_message'])

print(rewards)
