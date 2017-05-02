import numpy as np
import pandas as pd
import gym
import h5py
import random
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam

WEIGHT_SAVE = './config/weights.h5'
MODEL_SAVE_JSON = './config/model.txt'
CSV_FILE = './config/plot_data.csv'
EARLY = 10
EPOCH = 500


class Learner:
    def __init__(self, environment, eta=0.001, epsilon=1,
                 epsilon_decay=0.9995, epsilon_min=0.01,
                 train_start=1000, gamma=0.99, batch_size=64):
        # Environment data
        self.environment = environment
        self.input_shape = environment.observation_space.shape
        self.output_num = environment.action_space.n
        self.memory = []

        # Use queue instead of normal list, so we can always use better
        # training examples
        self.memory = deque(maxlen=2000)

        # Learning rate
        self.eta = eta

        # Start training after 1000 (building memory)
        self.train_start = train_start
        self.batch_size = batch_size

        # Exploration configuration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Discounted reward
        self.gamma = gamma

        # Init model
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.update_target_model()

    def init_model(self):
        """ Use Keras to build a model for q-learning."""
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.input_shape[0],
                        activation='relu'))
        model.add(Dense(units=24, activation='relu'))
        # We want rewards instead of probability, so use linear here
        model.add(Dense(units=self.output_num, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.eta))
        return model

    def remember_play(self, state, action, reward, next_state, done):
        """ Record the previous plays."""
        # Use tuple to represent one play
        self.memory.append((state, action, reward, next_state, done))

        # Decrease epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # Copy the weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        """ Try `batch_size` more moves on the current state, we use game
            data from `self.memory`, so they are labeled. New rewards (future
            rewards) are counted with a discount (self.gamma).
        """
        # Start only have enough memories
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))

        # Use mini_batch, sampling form the memory
        mini_batch = random.sample(self.memory, batch_size)

        # Since we are suing batch, we need to collect input and target
        input_update = np.zeros((batch_size, self.input_shape[0]))
        target_update = np.zeros((batch_size, self.output_num))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            # Add future discounted reward
            if not done:
                # Use target_model here, because we want to keep the weights
                # not changing in one complete game
                target[action] = reward + self.gamma * \
                        np.amax(self.target_model.predict(next_state)[0])
            else:
                target[action] = reward

            # Record the info into batch collection
            input_update[i] = state
            target_update[i] = target

        # Update model (also use a batch)
        self.model.fit(input_update, target_update, batch_size=batch_size,
                       epochs=1, verbose=0)

    def act(self, state):
        """ Pick an action based on either randomness(early stage), or model
            prediction. `self.epsilon` should shrink outside of this method.
        """
        # Random
        if np.random.rand() <= self.epsilon:
            return self.environment.action_space.sample()

        # Model prediction
        return np.argmax(self.model.predict(state)[0])


def train(epoch, rewards=1, punishment=-100):
    """ Use Learner to train an agent to play cartpole.

        Argument:
            rewards, the reward defaults to 1 in OpenAI, makes it tunable.
            punishment, the punishment defaults to -1 in OpenAI, we can tune
                        this parameter.
            batch_size, length of the subset of memory used to update model
    """
    # Init setting
    environment = gym.make('CartPole-v1')
    agent = Learner(environment)

    # Early stopping
    perfect_times = 0

    # Plot
    scores, epsilons = [], []

    for e in range(epoch):
        # Reset state for each epoch
        state = environment.reset().reshape((1, 4))
        done = False

        # Assume 2000 is our ultimate goal (cart keeps 2000 frames)
        for frame in range(2000):
            # Make one action
            action = agent.act(state)
            next_state, _, done, _ = environment.step(action)
            next_state = next_state.reshape((1, 4))

            # Customised reward and punishment
            reward = punishment if done else rewards

            # Build memory
            agent.remember_play(state, action, reward, next_state, done)

            # Train process
            agent.replay()
            state = next_state

            # End this game if done
            if done:
                # Update the target model for next inner prediction
                agent.update_target_model()

                # Store the scores for plotting
                scores.append(frame)
                epsilons.append(agent.epsilon)

                print(("epoch: {}/{}, score {}, " +
                      "epsilon {}").format(e, epoch, frame, agent.epsilon))
                break

        # Early stopping when getting `EARLY` continuous perfect score
        if frame == 499:
            perfect_times += 1
            if perfect_times == EARLY:
                break
        else:
            perfect_times = 0

    # Save the model and weights
    save_weight(agent.model)
    save_model(agent.model)

    # Save plotting data
    df = pd.DataFrame()
    df['epoch'] = range(1, len(scores) + 1)
    df['score'] = scores
    df['epsilon'] = epsilons
    df.to_csv(CSV_FILE, index=False)

    return agent


def save_weight(model):
    """ Have trouble with saving the model and weights, manually save the
        weights, based on the (issue)[https://github.com/farizrahman4u/seq2seq
        /issues/129].
    """
    file = h5py.File(WEIGHT_SAVE, 'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight' + str(i), data=weight[i])
    file.close()


def load_weight(model):
    """ Have trouble with saving the model and weights, manually load the
        weights, based on the (issue)[https://github.com/farizrahman4u/seq2seq
        /issues/129].
    """
    file = h5py.File(WEIGHT_SAVE, 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)


def save_model(model):
    """ Have trouble with saving the model and weights, use json to store
        the model.
    """
    json_string = model.to_json()
    with open(MODEL_SAVE_JSON, 'w') as fp:
        fp.write(json_string)


def load_model():
    """ Have trouble with saving the model and weights, use json to load
        the model.
    """
    with open(MODEL_SAVE_JSON, 'r') as fp:
        json_string = fp.read()
        model = model_from_json(json_string)
        return model


def playgame(trained_model, how_many_gameplays):
    """ Render the game play using final trained model. Print out final
        average score.
    """
    env = gym.make('CartPole-v0')
    scores = []
    for each_game in range(how_many_gameplays):
        score = 0
        state = env.reset().reshape((1, 4))
        for _ in range(200):
            env.render()
            action = np.argmax(trained_model.predict(state))
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, 4))
            score += reward

            if done:
                break
            # Move to next state
            state = next_state

        # record the score for each game
        scores.append(score)

    print('Average Score', sum(scores) / len(scores))


def main():
    model = train(EPOCH, punishment=-100)
    playgame(model.model, 20)


if __name__ == '__main__':
    main()
