import gym
import collections
from statistics import mean, median
import numpy as np

env = gym.make('CartPole-v0')
random_games = 20000
maximum_steps = 500
minimum_score = 90


def initial_data():
    accepted_scores = []
    game_memory = []
    training_data = []
    for each_game in range(random_games):
        env.reset()
        scores = []
        score = 0
        previous_observation = []

        for _ in range(maximum_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            score += reward
            previous_observation = observation
            if done:
                break

        if score > minimum_score:
            accepted_scores.append(score)
            for data in game_memory:
                output = []
                action_taken = data[1]

                for action_index in range(env.action_space.n):
                    if action_index == action_taken:
                        output.append(1)
                    else:
                        output.append(0)

                training_data.append([data[0], output])

        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved_data.npy', training_data_save)

    print('Averaged accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(collections.Counter(accepted_scores))
    return training_data

initial_data()
