import numpy as np

class QLearning:
    def __init__(self, environment, learning_rate, discount_factor, exploration_prob, epochs, epsilon):
        self.env = environment
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon = epsilon

        observation_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        if len(observation_shape) == 1 and len(action_shape) == 1:
            q_table_shape = (observation_shape[0], action_shape[0])
        else:
            q_table_shape = (np.prod(observation_shape), action_shape[0])

        # Initialize Q-table with zeros
        self.q_table = np.zeros(q_table_shape)

    def learn(self):
        for epoch in range(self.epochs):
            state = self.env.reset()  # Reset environment and get initial state

            total_reward = 0  # For tracking total reward in this epoch

            while not self.env.terminated:
                # Choose action based on epsilon-greedy strategy
                if np.random.rand() < self.exploration_prob:
                    action = np.random.choice(len(self.env.action_space))  # Random action
                else:
                    action = np.argmax(self.q_table[state])  # Greedy action

                # Simulate the environment (move to the next state)
                next_state, reward, terminated = self.env.step(action)

                # Update Q-value using the Q-learning update rule
                self.q_table[state, action] += self.learning_rate * (
                        reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action])

                total_reward += reward  # Accumulate reward

                state = next_state  # Update current state

            print(f"Epoch {epoch + 1}/{self.epochs}, Total Reward: {total_reward}")

        # After training, the Q-table represents the learned Q-values
        print("Training completed.")
        print("Learned Q-table:")
        print(self.q_table)