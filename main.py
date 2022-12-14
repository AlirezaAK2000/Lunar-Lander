import gym
from dqn import Agent
from utils import plotLearning
import numpy as np

if __name__ == "__main__":
    
    env = gym.make('LunarLander-v2')
    agent = Agent(
        gamma= 0.99, epsilon=1.0, batch_size=256, n_action=4,
        eps_end= 0.01, input_dims=[8], lr=0.0001, eps_dec= 5e-4
    )
    
    scores, eps_history = [], []
    n_games = 500

    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)            
            
            agent.learn(steps)
            observation = observation_
            steps += 1
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    
    x = [i + 1 for i in range(n_games)]
    file_name = 'learning_plot.png'
    plotLearning(x, scores, eps_history, file_name)
    