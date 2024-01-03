import gymnasium as gym
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def load_env():
    env = gym.make("LunarLander-v2", render_mode="human")
    return env

def get_state(env):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    return np.reshape(state, (1, 8))

# Modeli yükle
model_name = "a_50olum_1000.h5"
model = load_model(model_name)

# Ortamı yükle
env = load_env()

num_episodes = 50
max_steps = 2000
episode_rewards = []  # Her bölüm için toplam ödülleri saklamak için liste

for episode in range(num_episodes):
    state = get_state(env)
    total_reward = 0
    count = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(model.predict(state)[0])
        result = env.step(action)
        next_state, reward, done, _, _ = result
        next_state = np.reshape(next_state, (1, 8))
        total_reward += reward
        state = next_state
        count=count+1
        print(count)
        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode: {episode+1}, Total reward: {total_reward}")

env.close()

# Görselleştirme
plt.plot(range(1, num_episodes + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()
