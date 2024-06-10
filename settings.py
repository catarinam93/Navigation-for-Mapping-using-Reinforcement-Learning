'''This file contains a set of reward constants used in a reinforcement learning framework for a robot. 
These constants define the rewards or penalties that the robot receives based on its actions and 
outcomes during training or operation. These rewards are crucial for shaping the robot's 
behavior and guiding it towards achieving its goals.'''

# Reward received when no significant event occurs. Typically used as a default value.
NULL_REWARD = 0.0

# Reward received upon successfully completing a task or reaching the final goal.
FINAL_REWARD = 1000.0

# Reward given for neutral actions or states that do not significantly advance or hinder progress.
NEUTRAL_REWARD = 100.0

# Penalty incurred for actions or states that hinder progress or are undesirable.
PENALTY_REWARD = -500.0

# Severe penalty received for collisions, encouraging the robot to avoid obstacles and unsafe behaviors.
COLLISION_REWARD = -900.0
