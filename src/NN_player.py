import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def set_model(input_size, output_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, learning_rate=0.001, name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, output_size, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = set_model(input_size=len(X[0]), output_size=output_size)

    model.fit({'input': X}, {'targets': Y}, n_epoch=3, run_id='openAI')
    model.save('saved_model.model')
    return model


def playgame(env, maximum_steps, trained_model, action_choices, acceptable_score, how_many_gameplays):
    scores = []
    choices = []
    training_data = []
    for each_game in range(how_many_gameplays):
        score = 0
        game_memory = []
        previous_observation = []
        env.reset()

        for _ in range(maximum_steps):
            env.render()
            if len(previous_observation) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(
                    trained_model.predict(previous_observation.reshape(-1, len(previous_observation), 1))[0])

            choices.append(action)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            observation, reward, done, info = env.step(action)
            previous_observation = observation
            score += reward

            if done:
                break

        # record the score for each game
        scores.append(score)

        # the training data generated here can be for future use, but I am not using it right now
        if score > acceptable_score:
            for data in game_memory:
                output = []
                action_taken = data[1]

                for action_index in range(env.action_space.n):
                    if action_index == action_taken:
                        output.append(1)
                    else:
                        output.append(0)

                training_data.append([data[0], output])

    print('Average Score', sum(scores) / len(scores))
    # analyse the distribution of each action chosen
    for _ in range(action_choices):
        print("Choice", _, ":", choices.count(_) / len(choices))
    return training_data
