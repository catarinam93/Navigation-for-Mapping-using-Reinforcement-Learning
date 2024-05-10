import numpy as np

class QLearning:
    def __init__(self, environment, learning_rate, discount_factor, exploration_prob, epochs, epsilon):
        self.env = environment
        self.epochs = epochs
        self.q_table = np.zeros((len(self.env.observation_space), len(self.env.action_space))) # Initialize Q-table with zeros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon = epsilon

    def learn(self):
        for epoch in range(self.epochs):
            self.env.reset()  # Reset environment

            while not self.env.terminated:
            # ESTA PARTE PRECISA DE SER AJUSTADA, PLEEEEASE VÊ ISTO E ACHO QUE TEMOS O MODELO DO QLEARNING A TREINAR
            '''    # Choose action with epsilon-greedy strategy
                if np.random.rand() < self.exploration_prob:
                    action = np.random.randint(0, len(self.env.action_space))  # Explore
                else:
                    action = np.argmax(self.q_table[current_state])  # Exploit'''

                # Simulate the environment (move to the next state)
                next_state, reward, terminated = self.env.step(action)

                # Update Q-value using the Q-learning update rule
                self.q_table[current_state, action] += self.learning_rate * \
                    (reward + self.discount_factor *
                     np.max(self.q_table[next_state]) - self.q_table[current_state, action])


        # After training, the Q-table represents the learned Q-values
        print("Learned Q-table:")
        print(self.q_table)
