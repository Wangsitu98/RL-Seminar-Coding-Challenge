# RL-Seminar-Coding-Challenge
A small coding challenge of deep RL seminar at ETHZ

I used A3C to implement an agent playing battleship. With 5 threads and about 300 episodes of training the agent was able to achieve a score of about 12 (random action would give 4), which leads the agent to win the game. The variance is low as shown below in 'solution' folder im images 'result0', 'result1' and 'result2'.

The implementation of the code is also in 'solution' folder and one can run 'main.py' to see the training process. In 'config' one can change the number of threads used or whether to load the previous model.

The overall A3C implementation is adapted from (but heavily amended) https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2 by Arthur Juliani

