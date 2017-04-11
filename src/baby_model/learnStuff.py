'''
Created on Apr 8, 2017

@author: Brandon
'''
import gym
import random
import numpy as np
import keras
import tensorflow
from statistics import mean, median
from collections import Counter

learnRate = .001

env = gym.make('Breakout-v0')
env.reset()
maxActions = 5000

numRandomGames = 500
randomScoreThreshold = 25


def generateRandomGames():
    trainingData = [] #store data pairwise as {Observation, move}
    scores = [] #stores scores of sample data
    acceptedScores = [] #scores we are keeping for learning
    
    #create randomly generated games
    for _ in range(numRandomGames):
        env.reset()
        score = 0
        gameMemory = [] #moves made in this game and details of the state
        previousObservation = [] #current board state
        #go through max actions possible in a game
        for _ in range(maxActions):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            #save actions correlated to input
            if len(previousObservation) > 0 :
                gameMemory.append([previousObservation, action])
            previousObservation = observation
            score += reward
            if done:
                break
            
            #if exceeds tgreshhold keep game
        if score >= randomScoreThreshold:
            acceptedScores.append(score)
            #Convert action to 1 hot TODO: Change to 1 hot interpretation of 1 hot
            for data in gameMemory:
                print(data[1])
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                trainingData.append([data[0], output])
        scores.append(score)
        
    # random stats
    print('Average accepted score:',mean(acceptedScores))
    print('Median score for accepted scores:',median(acceptedScores))
    print(Counter(acceptedScores))
    
    return trainingData

generateRandomGames();