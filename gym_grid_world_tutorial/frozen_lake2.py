from contextlib import closing
from enum import Enum
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Action(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def text2action(text: str) -> Action:
    if text == "LEFT":
        return Action.LEFT
    if text == "DOWN":
        return Action.DOWN
    if text == "RIGHT":
        return Action.RIGHT
    if text == "UP":
        return Action.UP
    raise ValueError(f"Unknown action: {text}")

def action2text(action: Action) -> str:
    if action == Action.LEFT:
        return "LEFT"
    if action == Action.DOWN:
        return "DOWN"
    if action == Action.RIGHT:
        return "RIGHT"
    if action == Action.UP:
        return "UP"
    raise ValueError(f"Unknown action: {action}")

def actionId2text(action: int) -> str:
    actions = ["LEFT", "DOWN", "RIGHT", "UP"]
    try:
        return actions[action]
    except IndexError:
        raise ValueError(f"Unknown action: {action}")

def actionText2id(action: str) -> int:
    return text2action(action).value

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H" and board[r_new][c_new] != "B":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H", "B"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class FrozenLake2Env(Env):
    """
    Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H)
    by walking over the Frozen(F) lake and being unable to cross Blocked tiles (B).
    The agent may not always move in the intended direction due to the slippery nature of the frozen lake.


    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Rewards

    Reward schedule:
    - Reach goal(G): +1
    - Reach hole(H): -1
    - Reach frozen(F): 0

    ### Arguments

    ```
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc`: Used to specify custom map for frozen lake. For example,

        desc=["SFFF", "FHFH", "FFFH", "HFFG"].

        A random generated map can be specified by calling the function `generate_random_map`. For example,

        ```
        from gym.envs.toy_text.frozen_lake import generate_random_map

        gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
        ```

    `map_name`: ID to use any of the preloaded maps.

        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]

    `is_slippery`: True/False. If True will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.

        For example, if action is left and is_slippery is True, then:
        - P(move left)=1/3
        - P(move up)=1/3
        - P(move down)=1/3

    ### Version History
    * v2: Added blocked tile
    * v1: Bug fixes to rewards
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def cell_id_to_cell_coordinates(self,s):
        return s // self.ncol, s % self.ncol

    def cell_type_from_cell_coordinates(self,s):
        row,col = self.cell_id_to_cell_coordinates(s)
        return self.desc[row,col]
    
    def cell_coordinates_to_cell_id(self,row,col):
        return row * self.ncol + col
    
    def is_blocked_coordinates(self,row,col):
        return self.desc[row,col] == b"B"

    def transition_coordinates(self, row, col, action):
        if action == LEFT:
            new_row = row
            new_col = max(col - 1, 0)
        elif action == DOWN:
            new_row = min(row + 1, self.nrow - 1)
            new_col = col
        elif action == RIGHT:
            new_row = row
            new_col = min(col + 1, self.ncol - 1)
        elif action == UP:
            new_row = max(row - 1, 0)
            new_col = col
        return (new_row, new_col)
    
    def scale_value_in01(self,value,minmax_values):
        min_value, max_value = minmax_values
        return (value - min_value) / (max_value - min_value)
    
    def getColor01(self,val, minmax):
        minVal, maxVal = minmax
        r, g = 0.0, 0.0
        if val < 0 and minVal < 0:
            r = val * 0.65 / minVal
        if val > 0 and maxVal > 0:
            g = val * 0.65 / maxVal
        return (r,g,0.0)
    
    def getColor0255(self,val, minmax):
        r,g,b = self.getColor01(val, minmax)
        return (int(r*255),int(g*255),int(b*255))

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=False,
        action_success_rate=1.0
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        (self.nrow, self.ncol) = (nrow, ncol) = desc.shape
        self.reward_range = (-1, 1)

        assert 0.0 <= action_success_rate <= 1.0, "action_success_rate must be in [0, 1]"
        assert isinstance(action_success_rate, float), "action_success_rate must be a float"

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        if is_slippery:
            action_success_rate = 1.0/(nA-1.0)

        self.p_success = action_success_rate
        self.p_fail = (1.0 - action_success_rate) / (nA-1.0)
        

        def letter_from_state(s):
            return self.cell_type_from_cell_coordinates(s)

        def to_s(row, col):
            return self.cell_coordinates_to_cell_id(row,col)

        def inc(row, col, a):
            return self.transition_coordinates(row, col, a)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = 0.0
            if bytes(newletter) in b"BF":
                reward=0.0
            if bytes(newletter) == b"H":
                reward = -1.0
            if bytes(newletter) == b"G":
                reward = 1.0
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if 1.0 != action_success_rate:
                            t_prob = []
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                if a == b:
                                    p = self.p_success
                                else:
                                    p = self.p_fail
                                ns,r,t = update_probability_matrix(row, col, b)
                                if letter_from_state(ns) != b"B":
                                    t_prob.append(
                                        (p, ns, r, t)
                                    )
                                # else:
                                #     t_prob.append(
                                #         (0.0, ns,r,t)
                                #     )
                            sum_prob = sum([prob for prob, _, _, _ in t_prob])
                            t_prob = [(prob / sum_prob, ns, r, t) for prob, ns, r, t in t_prob]
                            li.extend(t_prob)
                        elif 1.0 == action_success_rate:
                            ns,r,t = update_probability_matrix(row, col, a)
                            if letter_from_state(ns) != b"B":
                                li.append((1.0, ns, r, t))
                            else:
                                # it stays in the same state
                                ns = s
                                li.append((1.0, ns, r, t))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.state_space = self.observation_space

        self.render_mode = render_mode

        # pygame utils
        self.init_window()
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.blocked_by_warning_img = None

        # Create arrow tiles
        self.arrow_tiles = {
            Action.LEFT: "←",
            Action.DOWN: "↓",
            Action.RIGHT: "→",
            Action.UP: "↑",
        }

    def init_window(self,cell_side : int=64,max_win_side: int=1024):
        self.cell_side = cell_side
        self.max_win_side = max_win_side
        self.window_size = (min(cell_side * self.ncol, max_win_side), min(cell_side * self.nrow, max_win_side))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}
    
    def is_terminal_l(self,letter):
        return letter in b"GH"
    
    def is_terminal(self,s):
        return self.is_terminal_l(self.cell_type_from_cell_coordinates(s))
    
    def is_terminal_coordinates(self,row,col):
        return self.is_terminal_l(self.desc[row,col])
    
        
    def create_image_with_row_col_coordinates(self, res : np.ndarray) -> np.ndarray:
        """
        Create an image from the grid representation adding a row and a column
        where in the first row the column index is displayed and in the first column the row index is displayed.
        It uses the cell_side and max_win_side attributes to calculate the size of the window, so if you have made
        changes to these attributes since you produced res, you have to set them back to the values they had when you
        produced res.
        """
        import pygame
        #enlarge the image to add the row and column indexes
        self.init_window(self.cell_side,self.max_win_side)
        # add one row size and one column size to the window size
        self.window_size = (self.window_size[0] + self.cell_size[0], self.window_size[1] + self.cell_size[1])
        rows = self.nrow+1
        cols = self.ncol+1
        pygame.init()
        # Copy the res image to the window_surface translated by the cell size in both directions
        self.window_surface = pygame.Surface(self.window_size)
        self.window_surface.blit(pygame.surfarray.make_surface(np.transpose(res, axes=(1, 0, 2))), (self.cell_size[0], self.cell_size[1]))
        # Add the row and column indexes with black text color on white background
        ## Add the white backgrounds
        fill_color = (255, 255, 255)  # White
        for row in range(0,rows):
            pos = (0, row * self.cell_size[1])
            rect = (*pos, *self.cell_size)
            pygame.draw.rect(self.window_surface, fill_color, rect)
        for col in range(1,cols):
            pos = (col * self.cell_size[0], 0)
            rect = (*pos, *self.cell_size)
            pygame.draw.rect(self.window_surface, fill_color, rect)
        ## Add the row and column labels
        text_color = (0, 0, 0)  # Black
        font = self.get_font(30)
        for row in range(1,rows):
            pos = (0, row * self.cell_size[1])
            rect = (*pos, *self.cell_size)
            pyrect = pygame.Rect(*rect)
            text = font.render(str(row-1), True, text_color)
            text_rect = text.get_rect(center=self._center_small_rect(rect, text.get_size()))
            text_rect.center = pyrect.center
            self.window_surface.blit(text, text_rect)
        for col in range(1,cols):
            pos = (col * self.cell_size[0], 0)
            rect = (*pos, *self.cell_size)
            pyrect = pygame.Rect(*rect)
            text = font.render(str(col-1), True, text_color)
            text_rect = text.get_rect(center=self._center_small_rect(rect, text.get_size()))
            text_rect.center = pyrect.center
            self.window_surface.blit(text, text_rect)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))



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
        
    def render_with_coordinates(self):
        res = self.render()
        return self.create_image_with_row_col_coordinates(res)

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
        if self.blocked_by_warning_img is None:
            file_name = path.join(path.dirname(__file__), "img/warning_slippery_ice.png")
            self.blocked_by_warning_img = pygame.transform.scale(
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
                elif desc[y][x] == b"B":
                    self.window_surface.blit(self.blocked_by_warning_img, pos)
                
                #print(f"{desc[y][x]} ",end="")
                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)
            #print("")

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

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        """
        Given a big rectangle and the dimensions of a smaller rectangle, return the coordinates of the top-left corner
        of the smaller rectangle such that it is centered inside the big rectangle.
        """
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    # Function to dynamically calculate font size
    def get_font(self,size):
        import pygame
        return pygame.font.Font(None, size)
    
    def create_value_image(self, values, minmax_values : tuple[float,float]) -> np.ndarray:
        """
        Create an image of the value function as a grid with the value number
        in the center of each cell and the cell colored according to the value.
         - In red you have all terminal states that end in failure.
         - In grey you have all the cells containing an obstacle without any value in them.
         - In most vivid green you have the cells with terminal states that end in success.
         - In shades of green from light to most vivid you have the cells with the 
         lowest values (most light green) to the highest values (most vivid green).

            Params
            ------
            values : a function that maps a state: a couple of coordinates (row, col) to a value
            minmax_values : a tuple with the minimum and maximum values of the value function

            Returns
            -------
            An image of the value function as a grid.
        """
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        self.init_window(128,2048)
        pygame.init()
        self.window_surface = pygame.Surface(self.window_size)
        
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        min_cell_size : int = min(self.cell_size[0],self.cell_size[1])
        font_size : int = int(min_cell_size * 0.5)
        font = self.get_font(font_size)

        cell_outline_color = (255, 255, 255)  # White
        text_color = (255, 255, 255)  # White

        # Iterate over the cells and draw the value
        desc = self.desc.tolist()
        for row in range(self.nrow):
            for col in range(self.ncol):
                pos = (col * self.cell_size[0], (row) * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                pyrect = pygame.Rect(*rect)
                if desc[row][col] == b"B":
                    pygame.draw.rect(self.window_surface, (128, 128, 128), pyrect)
                    #print(f"({row},{col}) : ???, ???? : letter: {desc[row][col]}")
                    pygame.draw.rect(self.window_surface, cell_outline_color, pyrect, width=1)
                else:
                    value = values(row, col)
                    str_value = f"{value:.2f}"
                    #print(f"({row},{col}) : {value}, {str_value} : letter: {desc[row][col]}",end=" ")
                    font = pygame.font.Font(None, 30)
                    text = font.render(str_value, True, text_color)
                    text_rect = text.get_rect(center=self._center_small_rect(rect, text.get_size()))
                    text_rect.center = pyrect.center
                    terminal_sizes = tuple(np.subtract(self.cell_size,(10,10)))
                    terminal_rect = pygame.Rect(*self._center_small_rect(rect, terminal_sizes), *terminal_sizes)
                    if desc[row][col] in b"GH":
                        if desc[row][col] == b"H":
                            pygame.draw.rect(self.window_surface, (255, 0, 0), pyrect)
                        elif desc[row][col] == b"G":
                            pygame.draw.rect(self.window_surface, (0, 255, 0), pyrect)
                        pygame.draw.rect(self.window_surface, cell_outline_color, terminal_rect, width=2)
                    else:
                        color = self.getColor0255(value,minmax_values)
                        #print(f'value: {value}, color: {color}')
                        pygame.draw.rect(self.window_surface, color, pyrect)
                    pygame.draw.rect(self.window_surface, cell_outline_color, pyrect, width=1)
                    self.window_surface.blit(text, text_rect)
            #print("")
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))
    
    def create_value_image_with_row_col_coordinates(self, values, minmax_values : tuple[float,float]) -> np.ndarray:
        """
        Use the create_value_image function to create an image of the value function in the grid and add a row and a column
        where in the first row the column index is displayed and in the first column the row index is displayed.
        """
        res = self.create_value_image(values,minmax_values)
        return self.create_image_with_row_col_coordinates(res)
  
    def create_policy_image(self, policy) -> np.ndarray:
        """
        Create an image of the policy as a grid with an arrow in the center of each cell
        pointing in the direction of the action to take.
        """
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        self.init_window(128,2048)
        pygame.init()
        self.window_surface = pygame.Surface(self.window_size)
        
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        min_cell_size : int = min(self.cell_size[0],self.cell_size[1])
        font_size : int = int(min_cell_size * 0.5)
        font = pygame.font.SysFont("Arial", font_size)

        cell_outline_color = (255, 255, 255)    # White
        arrow_color = (255, 255, 255)            # White

        # Use the arrow symbols stored in the arrow_tiles attribute
        # to draw text in the center of the cell

        # Iterate over the cells and draw the policy
        desc = self.desc.tolist()
        for row in range(self.nrow):
            for col in range(self.ncol):
                pos = (col * self.cell_size[0], (row) * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                pyrect = pygame.Rect(*rect)
                if desc[row][col] == b"B":
                    pygame.draw.rect(self.window_surface, (128, 128, 128), pyrect)
                    pygame.draw.rect(self.window_surface, cell_outline_color, pyrect, width=1)
                else:
                    action = policy(row, col)
                    if action is not None:
                        # get character for action
                        arrow = self.arrow_tiles[Action(action)]
                        text = font.render(arrow, True, arrow_color)
                        text_rect = text.get_rect(center=self._center_small_rect(rect, text.get_size()))
                        text_rect.center = pyrect.center
                        pygame.draw.rect(self.window_surface, cell_outline_color, pyrect, width=1)
                        self.window_surface.blit(text, text_rect)
                        
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))

    def create_policy_image_with_row_col_coordinates(self, policy) -> np.ndarray:
        """
        Use the create_policy_image function to create an image of the policy in the grid and add a row and a column
        where in the first row the column index is displayed and in the first column the row index is displayed.
        """
        res = self.create_policy_image(policy)
        return self.create_image_with_row_col_coordinates(res)

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/