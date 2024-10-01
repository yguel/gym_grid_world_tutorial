import gymnasium as gym
from gymnasium import RewardWrapper

# Change the reward function to be -1 for holes
# Final reward function is:
#  * -1 for holes
#  *  1 for the goal
#  *  0 for all other states

class CustomRewardWrapper(RewardWrapper):
    """
    Allows to make reward be -1 for a hole
    """
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Modify the reward."""
        # Check the agent's current state
        if b'H' == self.env.unwrapped.desc[self.env.s // self.env.ncol][self.env.s % self.env.ncol]:  # 'H' represents a hole
            return -1  # Set reward to -1 if the agent is in a hole
        return reward  # Otherwise, keep the original reward

# Custom Frozen Lake environment with Blocked "B" states
class CustomFrozenLakeEnv(gym.Env):
    def __init__(self, custom_map):
        # Initialize the Frozen Lake environment with custom map
        self.env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False)
        self.custom_map = custom_map
        self.env.reset()

        # Modify transitions to account for Blocked "B" states
        self.modify_transitions()

    def modify_transitions(self):
        """
        Modify the transition probabilities to account for blocked 'B' states.
        """
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                # Extract original transitions for each (state, action)
                transitions = self.env.P[state][action]
                new_transitions = []

                for prob, next_state, reward, done in transitions:
                    # Find the coordinates of the next state in the grid
                    row, col = divmod(next_state, self.env.ncol)

                    # Check if the next state is a Blocked 'B' state
                    if self.custom_map[row][col] == b'B':
                        # Prevent movement into the 'B' state by keeping the agent in the same state
                        new_transitions.append((prob, state, 0, False))  # No reward, not done
                    else:
                        new_transitions.append((prob, next_state, reward, done))

                # Update transitions
                self.env.P[state][action] = new_transitions

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()



#########
# Usage #
#########

# Create a custom map with "B" for blocked states
custom_map = [
    b"FFFG",
    b"FBFH",
    b"SFFF"
]
# Initialize the custom environment with the blocked states
custom_env = CustomFrozenLakeEnv(custom_map)

# Wrap the environment with the custom reward wrapper
env = CustomRewardWrapper(custom_env)

# Example of running the environment
observation = env.reset()