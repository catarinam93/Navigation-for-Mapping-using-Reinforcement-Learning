''' Q-LEARNING ALGORITHM - the base of this algorithm was taken from https://www.geeksforgeeks.org/q-learning-in-python/'''

import numpy as np

class QLearning:
    def __init__(self, environment, learning_rate, discount_factor, exploration_prob, epochs, epsilon):
        self.env = environment
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon = epsilon

        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        if len(self.observation_shape) == 1 and len(self.action_shape) == 1:
            q_table_shape = (self.observation_shape[0], self.action_shape[0])
        else:
            q_table_shape = (np.prod(self.observation_shape), self.action_shape[0])

        # Initialize Q-table with zeros
        self.q_table = np.zeros(q_table_shape)

    def learn(self):
        for epoch in range(self.epochs):
            current_state = self.env.reset()  # Reset environment
            total_reward = 0  # For tracking total reward in this epoch

            while not self.env.terminated:
                # Choose action with epsilon-greedy strategy
                if np.random.rand() < self.exploration_prob:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[current_state])  # Exploit

                next_state, reward, terminated = self.env.step(action) # Simulate the environment (move to the next state)

                # Update Q-value using the Q-learning update rule
                self.q_table[current_state, action] += self.learning_rate * \
                    (reward + self.discount_factor *
                     np.max(self.q_table[next_state]) - self.q_table[current_state, action])

                total_reward += reward  # Accumulate reward
                current_state = next_state # Change do the next state

            print(f"Epoch {epoch + 1}/{self.epochs}, Total Reward: {total_reward}")

        # After training, the Q-table represents the learned Q-values
        print("Training completed.")
        print("Learned Q-table:")
        print(self.q_table)
