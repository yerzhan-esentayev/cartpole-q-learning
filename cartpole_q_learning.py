import gym
import numpy as np
import random
from gym.wrappers import RecordVideo
import os

# Создаем окружение
env = gym.make("CartPole-v1")


# Дискретизация состояний
def discretize_state(state, bins):
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    state_high[1] = 5  # Ограничим скорость тележки
    state_low[1] = -5
    state_high[3] = 5  # Ограничим скорость палки
    state_low[3] = -5
    ratios = (state - state_low) / (state_high - state_low)
    new_state = (ratios * bins).astype(int)
    new_state = np.clip(new_state, 0, bins - 1)
    return tuple(new_state)


# Параметры
n_bins = 24
bins = np.array([n_bins] * len(env.observation_space.high))
q_table = np.random.uniform(low=-1, high=1, size=(n_bins, n_bins, n_bins, n_bins, env.action_space.n))

alpha = 0.1  # скорость обучения
gamma = 0.99  # коэффициент дисконтирования
epsilon = 1.0  # начальная вероятность случайного действия
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 5000

# Обучение
for episode in range(n_episodes):
    state = discretize_state(env.reset()[0], bins)
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state_raw, reward, done, info = env.step(action)
        next_state = discretize_state(next_state_raw, bins)

        # Обновляем Q-таблицу
        best_next_action = np.max(q_table[next_state])
        q_table[state + (action,)] += alpha * (reward + gamma * best_next_action - q_table[state + (action,)])

        state = next_state
        total_reward += reward

    # Понижаем вероятность случайного действия
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Выводим прогресс
    if episode % 100 == 0:
        print(f"Эпизод {episode}, Вознаграждение: {total_reward}, Эпсилон: {epsilon:.3f}")

# --- ТЕСТИРОВАНИЕ И ЗАПИСЬ ВИДЕО ---

# Пересоздаем окружение с записью видео
env.close()
env = gym.make("CartPole-v1")
video_dir = './video'

# Создаем папку для видео, если её нет
os.makedirs(video_dir, exist_ok=True)

# Обернуть окружение для записи видео
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True)

state = discretize_state(env.reset()[0], bins)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])

    next_state_raw, reward, done, info = env.step(action)
    next_state = discretize_state(next_state_raw, bins)
    state = next_state
    total_reward += reward

print(f"Тестовое вознаграждение: {total_reward}")

env.close()

print(f"Видео сохранено в папке {video_dir}")