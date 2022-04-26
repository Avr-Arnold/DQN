
import gym

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn import  neural_network as ANN
import pandas as pd

import tensorflow as tf

class DQN_LL():
    def __init__(self, episodes=10000, gamma=.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.95, alpha=0.001, batch_size=64):
        self.memory = np.zeros(19)
        self.episode_batch = np.zeros(19)
        self.mem_len = 500000
        self.mem_counter = 0
        self.env = gym.make('LunarLander-v2')
        self.is_fit = False
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        self.number_of_episodes = episodes
        tf.compat.v1.disable_eager_execution()
        tf.config.optimizer.set_jit(True)
        self.batch_size = batch_size
        self.env.seed(42)
        np.random.seed(42)

        #self.model = ANN.MLPRegressor((32, 32), warm_start=True)
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=8, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def remember(self, state, action, reward, next_state, done):

        m = np.concatenate((state, [action], [reward], next_state, [done])) #.hstack((np.hstack((np.hstack((np.hstack((state, action)), reward)), next_state)), done))
        #self.episode_batch = np.vstack((m, self.episode_batch))
        # m = np.hstack((np.hstack((np.hstack((np.hstack((state, action)), reward)), next_state)), done))
        if self.mem_counter < self.mem_len:
            #m = np.concatenate((state, [action], [reward], next_state, [done]))
            self.memory = np.vstack((self.memory,m))
        else:
            self.memory[self.mem_counter % self.mem_len] = m
        self.mem_counter += 1

    def choose_action(self, state):
        if (np.random.random() <= self.epsilon):
            return np.random.randint(4)

        return np.argmax(self.model.predict(self.process_sequence(state)))

    def process_sequence(self, state):
        return np.reshape(state, [1, 8])

    def unpack(self, row):
        state = row[:8]
        action = int(row[8])
        reward = row[9]
        next_state = row[10:18]
        done = row[18]
        return state, action, reward, next_state, done

    def batch_unpack(self, batch):
        state = batch[:, :8]
        action = batch[:, 8]
        reward = batch[:, 9]
        next_state = batch[:, 10:18]
        done = batch[:, 18]
        return state, action, reward, next_state, done

    def replay(self, batch_size):
        # X = np.zeros(8)
        # Y = np.zeros(4)

        batch = None
        if self.memory.shape[0] < batch_size:
            index = np.random.choice(self.memory.shape[0], self.memory.shape[0], replace=False)
            batch = self.memory[index]
        else:
            index = np.random.choice(self.memory.shape[0], batch_size, replace=False)
            batch = self.memory[index]

        states, actions, rewards, next_states, done = self.batch_unpack(batch)
        s_prime = np.max(self.model.predict_on_batch(next_states), axis=1)
        #state = np.max(self.model.predict_on_batch(states), axis=1)
        Y_Values = (rewards + (self.gamma * (s_prime)) * (1 - done))

        Y_predict = self.model.predict_on_batch(states)

        Y_predict[[np.arange(batch.shape[0])], [actions.astype(int)]] = Y_Values

        self.model.fit(states, Y_predict, verbose=0)
        # self.X = np.vstack((state,self.X))
        # self.Y = np.vstack((Y, self.Y))
        # X = X[1:, :]
        # Y = Y[1:]
        #self.model.fit(self.X, self.Y)


    def run(self):

        self.scores = []
        self.mean_score = []
        self.epsilon_decay_list = []
        for episode in range(self.number_of_episodes):
            state = self.env.reset() #Initialize sequence s1=x1 and preprocessed sequence w1~ws1ðÞ

            done = False
            i = 0
            episode_rewards = 0
            step_num = 1
            action = self.choose_action(state)
            while not done:
                self.env.render()
                #if step_num % 4 == 0:#frame skipping

                action = self.choose_action(state) #With probability e select a random action a_t
                # if step_num > 900:
                #     action = 0#np.random.randint(4)
                next_state, reward, done, _ = self.env.step(action) #Execute action a_t in emulator and observe reward r_t ...and imagext1

                self.remember(state, int(action), reward, next_state, done) # Store transition s_t,a_t,r_t,s_t+1  in D
                state = next_state

                episode_rewards += reward
                step_num += 1

                if step_num % 4 == 0:
                    self.replay(self.batch_size)


            self.scores.append(episode_rewards)
            self.mean_score.append(np.mean(self.scores[-100:]))
            self.epsilon_decay_list.append(self.epsilon)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.mean_score[-1] > 200:
                break
        return self.scores, self.mean_score, self.epsilon_decay_list

    def run_trained(self):

        self.scores = []
        self.mean_score = []
        #self.epsilon_decay_list = []
        for episode in range(101):
            state = self.env.reset() #Initialize sequence s1=x1 and preprocessed sequence w1~ws1ðÞ
            done = False
            i = 0
            episode_rewards = 0
            step_num = 1

            while not done:

                action = np.argmax(self.model.predict(self.process_sequence(state))) #With probability e select a random action a_t

                next_state, reward, done, _ = self.env.step(action) #Execute action a_t in emulator and observe reward r_t ...and imagext1

                state = next_state

                episode_rewards += reward
                step_num += 1


            self.scores.append(episode_rewards)
            self.mean_score.append(np.mean(self.scores[-100:]))

        return self.scores, self.mean_score#, self.epsilon_decay_list

if __name__ == '__main__':
    #standard params
    alpha = 0.001
    epsilon = 1.0

    gamma = 0.99


    epsilon_min = .01

#Epsilon Decay



    training_episodes = 500
    epsilon_decay = 0.95
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'epsilon_decay_list': epsilon_decay_list, 'epsilon_decay': epsilon_decay}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('Dec96R2.csv')
    i = 0
    epsilon_decay = 0.94
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'epsilon_decay_list': epsilon_decay_list, 'epsilon_decay': epsilon_decay}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('Dec94R2.csv')
    i = 0

#Alpha
    alpha = 0.00025
    epsilon = 1.0
    epsilon_decay = 0.95
    gamma = 0.99
    training_episodes = 700
    epsilon_min = .01
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'alpha': alpha,
            'epsilon_decay_list': epsilon_decay_list}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('A00025.csv')
    i = 0
    alpha = 0.0025
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'alpha': alpha,
            'epsilon_decay_list': epsilon_decay_list}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('A0025.csv')

#Gamma
    alpha = 0.001
    epsilon = 1.0
    epsilon_decay = 0.95
    training_episodes = 700
    epsilon_min = .01
    i = 0
    gamma = 0.995
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'gamma': gamma,
            'epsilon_decay_list': epsilon_decay_list}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('G995.csv')
    i = 0

    gamma = 0.9
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'gamma': gamma,
            'epsilon_decay_list': epsilon_decay_list}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('G95.csv')

    gamma = 0.99
    training_episodes = 500
    epsilon_decay = 0.95
    epsilon_min = .01
    model = DQN_LL(episodes=training_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    rewards, last_rewards_mean, epsilon_decay_list = model.run()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean, 'epsilon_decay_list': epsilon_decay_list, 'epsilon_decay': epsilon_decay}
    df_A = pd.DataFrame(dict)
    df_A.to_csv('Best.csv')
    i = 0

    rewards, last_rewards_mean = model.run_trained()
    dict = {'rewards': rewards, 'last_rewards_mean': last_rewards_mean}
    df_T = pd.DataFrame(dict)
    df_T.to_csv('Trained.csv')
    i = 0
