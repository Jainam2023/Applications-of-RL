import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

reward_arr=[]
cum_reward=0
i=0

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    cum_reward+=reward
    if terminated or truncated:
        reward_arr.append(cum_reward)
        i+=1
        observation, info = env.reset()
        cum_reward=0

episodes=np.arange(i)
plt.plot(episodes,reward_arr)
env.close()
plt.show()