import gym
import generate_training_data as stupid_teacher
import NN_player

env = gym.make('CartPole-v0')
# how many random game we play
random_games = 30000
# maximum step until which we stop the game
maximum_steps = 500
# minimum score beyond which we consider as our training data
minimum_score = 70

our_data = stupid_teacher.initial_data(env, random_games, maximum_steps, minimum_score)
our_model = NN_player.train_model(training_data=our_data, output_size=env.action_space.n)
NN_player.playgame(env=env, maximum_steps=800, trained_model=our_model,
                   action_choices=env.action_space.n, acceptable_score=minimum_score, how_many_gameplays=10)
