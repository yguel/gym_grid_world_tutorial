import gymnasium as gym
from gymnasium import RewardWrapper
import pygame
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

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
class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, desc, is_slippery=False, render_mode='rgb_array', map_name=None ):
        super().__init__(desc=desc,is_slippery=is_slippery,render_mode=render_mode,map_name=map_name)

        # Now set nrow and ncol after the parent class has initialized
        self.nrow, self.ncol = self.desc.shape
        
        self.custom_map = desc
        self.blocked_image = pygame.image.load('/media/manu/manuLinux/code/sources.yguel/python/ia/reinforcement_learning/gym_grid_world_tutorial/assets/blocked_tile.png')  # Load the custom image for blocked cells
        self.reset()
        # print(f"(row,col) = ({self.nrow}, {self.ncol})")
        # print(f"Expected row= {len(self.custom_map)}")
        # print(f"Expected col= {len(self.custom_map[0])}")

        # Modify transitions to account for Blocked "B" states
        self.modify_transitions()

    def modify_transitions(self):
        """
        Modify the transition probabilities to account for blocked 'B' states.
        """
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                # Extract original transitions for each (state, action)
                transitions = self.P[state][action]
                new_transitions = []

                for prob, next_state, reward, done in transitions:
                    # Find the coordinates of the next state in the grid
                    row, col = divmod(next_state, self.ncol)

                    # Check if the next state is a Blocked 'B' state
                    if self.desc[row,col] == b'B':
                        # Prevent movement into the 'B' state by keeping the agent in the same state
                        new_transitions.append((prob, state, 0, False))  # No reward, not done
                    else:
                        new_transitions.append((prob, next_state, reward, done))

                # Update transitions
                self.P[state][action] = new_transitions
    
    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
    
    def render(self):        
        # Use the default rendering for non-blocked cells
        frame = super().render()  # Call the default rendering method

        self.blocked_img = pygame.transform.scale(self.blocked_image, self.cell_size)  # Resize the image to match the cell size

        # Add custom rendering for 'B' cells using Pygame
        for row in range(self.nrow):
            for col in range(self.ncol):
                # Get the cell type (S, F, H, G, or B)
                cell_type = self.desc[row, col]

                # Draw the blocked image if it's a 'B' cell
                if cell_type == b'B':
                    pos = (col * self.cell_size[0], row * self.cell_size[1])
                    rect = (*pos, *self.cell_size)
                    self.window_surface.blit(self.blocked_image, pos)
        
        pygame.display.flip()  # Update the display

    def close(self):
        if self.window is not None:
            pygame.quit()



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