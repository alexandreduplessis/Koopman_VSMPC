import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot as plt
from vs_utils import draw_centered_square, find_max_size_square_center_coordinates, loss_to_reward


class VsEnv(gym.Env):
    """ VsEnv class:
    - length: length of the matrix (int)
    - width: width of the matrix (int)
    - size: size of the square (int)
    - state: current state of the environment (python dict)
    - counter: number of steps (int)
    
    Idea:
    Simulates and controls the output of a camera that detects a square of size size centered on (x, y)
    """
    
    
    def __init__(self, length, width, goal=[None, None, None], initial=[None, None, None], max_steps=100):
        self.length = length
        self.width = width
        self.counter = 0
        self.info = {}
        if goal[0] is None:
            self.goal = {
                "abscisse": self._matrix_abscisse_to_observation_abscisse(self.length//2),
                "ordonnee": self._matrix_ordonnee_to_observation_ordonnee(self.width//2),
                "depth": 1.
            }
        else:
            self.goal = {
                "abscisse": goal[0],
                "ordonnee": goal[1],
                "depth": goal[2]
            }
        if initial[0] is None:
            self.state = {
                "abscisse": 1.,
                "ordonnee": 2.,
                "depth": 0.3,
            }
        else:
            self.state = {
                "abscisse": initial[0],
                "ordonnee": initial[1],
                "depth": initial[2]
            }
        self.reward = loss_to_reward(self._distance_to_goal(self.state))
        self.historic = [self.reward]
        self.max_steps = max_steps
        self.done = False
        self.observation_space = spaces.Dict(
            {
                "abscisse": spaces.Box(-10., 10., shape=(1,)),
                "ordonnee": spaces.Box(-10., 10., shape=(1,)),
                "depth": spaces.Box(0., 10., shape=(1,)),
            }
        )
        self.action_space = spaces.Dict(
            {
                "abscisse": spaces.Box(-1., 1., shape=(1,)),
                "ordonnee": spaces.Box(-1., 1., shape=(1,)),
                "depth": spaces.Box(0., .2, shape=(1,)),
            }
        )
        self.fig, self.ax = None, None

    def _distance_to_goal(self, observation):
        """ Compute the distance to the goal """
        matrix_goal = self.observation_to_matrix(self.goal)
        matrix_state = self.observation_to_matrix(observation)
        norm = np.linalg.norm(matrix_goal - matrix_state)
        return norm
    
    def _depth_to_size(self, depth):
        """ Convert a depth to a size """
        return int(depth*self.length//10)
    
    def _size_to_depth(self, size):
        """ Convert a size to a depth """
        return size*10./self.length
    
    def _matrix_abscisse_to_observation_abscisse(self, matrix_abscisse):
        """ Convert a matrix abscisse to an observation abscisse """
        return matrix_abscisse/self.length*10.
    
    def _observation_abscisse_to_matrix_abscisse(self, observation_abscisse):
        """ Convert an observation abscisse to a matrix abscisse """
        return int(observation_abscisse*self.length//10)
    
    def _observation_ordonnee_to_matrix_ordonnee(self, observation_ordonnee):
        """ Convert an observation ordonnee to a matrix ordonnee """
        return int(observation_ordonnee*self.width//10)
    
    def _matrix_ordonnee_to_observation_ordonnee(self, matrix_ordonnee):
        """ Convert a matrix ordonnee to an observation ordonnee """
        return matrix_ordonnee/self.width*10.
    
    def matrix_to_observation(self, matrix):
        """
        Convert the matrix to an observation.
        input: matrix (np.array)
        output: observation (python dict)
        """
        abscisse, ordonnee, size = find_max_size_square_center_coordinates(matrix)
        return {
            "abscisse": self._matrix_abscisse_to_observation_abscisse(abscisse),
            "ordonnee": self._matrix_ordonnee_to_observation_ordonnee(ordonnee),
            "depth": self.size_to_depth(size)
        }
    
    def observation_to_matrix(self, observation):
        """
        Convert the observation to a matrix.
        input: observation (python dict)
        output: matrix (np.array)
        """
        observation_abscisse, observation_ordonnee, depth = observation["abscisse"], observation["ordonnee"], observation["depth"]
        return draw_centered_square(self._observation_abscisse_to_matrix_abscisse(observation_abscisse), 
                                    self._observation_ordonnee_to_matrix_ordonnee(observation_ordonnee), 
                                    self._depth_to_size(depth), 
                                    self.length, 
                                    self.width)
    
    def _is_valid_action(self, action):
        """ Check if the action is valid """
        state_copy = self.state.copy()
        next_state = {
            "abscisse": self.state["abscisse"] + action["abscisse"],
            "ordonnee": self.state["ordonnee"] + action["ordonnee"],
            "depth": self.state["depth"] + action["depth"]
        }
        next_state_matrix = self.observation_to_matrix(next_state)
        return np.nonzero(next_state_matrix)[0].size == 0
    
    def step(self, action=None):
        """ Perform one step of the environment's dynamics. """
        if action is None:
            keep = True
            while keep:
                action = self.action_space.sample()
                keep = self._is_valid_action(action)
            self.info["action"] = np.array([action["abscisse"], action["ordonnee"], action["depth"]])
        # print("action:", [action["abscisse"], action["ordonnee"], action["depth"]])
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        assert self.done is False, "Episode is done, because:\n {}".format(self.info)
        assert self.counter <= self.max_steps, "Episode is done, because:\n {}".format(self.info)
        
        self.state = {
            "abscisse": self.state["abscisse"] + action["abscisse"],
            "ordonnee": self.state["ordonnee"] + action["ordonnee"],
            "depth": self.state["depth"] + action["depth"]
        }
        self.reward = loss_to_reward(self._distance_to_goal(self.state))
        print("Step {i}: {distance}".format(i=self.counter, distance=self.reward))
        if self.counter == self.max_steps:
            self.done = True
            reason = "The environment has reached its maximum number of steps."
            self.info["reason"] = reason
            print("The environment has reached its maximum number of steps.")
        elif self._distance_to_goal(self.state) < 0.1:
            self.done = True
            reason = "The environment has reached the goal."
            self.info["reason"] = reason
            print("The environment has reached the goal.")
        else:
            self.done = False
            self.counter += 1
        self.historic.append(self.reward)
        return self.state, self.reward, self.done, self.info
    
    def reset(self, x=1., y=1., depth=1.):
        """ Reset the state of the environment to an initial state. """
        self.state = {
            "abscisse": x,
            "ordonnee": y,
            "depth": depth
        }
        self.counter = 0
        self.done = False
        self.fig, self.ax = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']],
                              constrained_layout=True)
        plt.suptitle("VS Performance", fontsize=14)
        matrix_goal = self.observation_to_matrix(self.goal)
        self.ax['right'].imshow(matrix_goal, cmap='gray')
        self.ax['right'].set_title("Goal")
        self.ax['left'].set_title("Current")
        self.ax['bottom'].set_title("Reward")
        return self.state
    
    def current_state_matrix(self):
        """ Return the current state as a matrix """
        return self.observation_to_matrix(self.state)
    
    def render(self, display=None):
        """ Render the environment to the screen. """
        if display == "matrix":
            state_string = """
            Abscisse: {abscisse}
            Ordonnee: {ordonnee}
            Depth: {depth}
            """.format(**self.state)
            return self.state
        else:
            matrix = self.observation_to_matrix(self.state)
            self.ax['left'].imshow(matrix, cmap='gray')
            self.ax['bottom'].plot(self.historic)
            if self.done:
                plt.show()
            else:
                plt.pause(0.1)