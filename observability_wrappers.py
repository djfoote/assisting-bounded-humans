from typing import Tuple, Union

import gymnasium as gym
import numpy as np
from stealing_gridworld import StealingGridworld

from imitation.data.types import TrajectoryWithRew
from imitation_modules import ObservationFunction

from stable_baselines3.common.vec_env import VecEnv

########################################################################################################################                   
#                                           OBSERVABILITY FUNCTIONS                                                    #
########################################################################################################################


# PartialGridVisibility: This observation function masks the visibility of the agent in the gridworld environment.
# The visibility mask is a boolean matrix that is multiplied element-wise with the observation tensor. 
# The mask is applied to all channels except the last one, which is the number of carried pellets. 
# The mask is STATIC, meaning it does not change over time. 

# As for now, the visibility mask is hardcoded to be a n-1 x n-1 square in the center of the grid, thus leaving the borders invisible.

class PartialGridVisibility(ObservationFunction):

    # TODO (joan): Embed the visibility mask in the observation function
    # in order to make it more general and reusable. Also avoids having to pass it as an argument.
    # The possible visibility masks for now are the default one, and a camera-like one (under DynamicGridVisibility).

    def __init__(self, 
                 env: Union[gym.Env, VecEnv], 
                 mask_key: str = None, 
                 feedback: str = "scalar"):
        self.env = env
        self.grid_size = self.env.envs[0].grid_size
        self.visibility_mask = self.construct_visibility_mask(mask_key)
        self.feedback = feedback

    def construct_visibility_mask(self, 
                                  visibility_mask_key:str = "(n-1)x(n-1)") -> np.ndarray:
        # Any other visibility mask keys should be added here.
        if visibility_mask_key == "(n-1)x(n-1)":
            visibility_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
            visibility_mask[1:-1, 1:-1] = True
            return visibility_mask
        else:
            raise ValueError(f"Unknown visibility mask key {visibility_mask_key}.")

    def __call__(self, fragments):
        if self.feedback == "scalar":
            return self.process_scalar_feedback(fragments)
        elif self.feedback == "preference":
            return self.process_preference_feedback(fragments)

    def process_scalar_feedback(self, 
                                fragment: TrajectoryWithRew) -> TrajectoryWithRew:
        
        masked_obs = fragment.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis] # apply the visibility mask to all channels except the last one
        new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1) # concatenate the masked obs with the last channel
        agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
        new_rew = fragment.rews * agent_visible
        return TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew)

    def process_preference_feedback(self, 
                                    fragment_pair: Tuple[TrajectoryWithRew, TrajectoryWithRew]) -> Tuple[TrajectoryWithRew, TrajectoryWithRew]:
        
        processed_fragments = []

        for fragment in fragment_pair:
            masked_obs = fragment.obs[:, :-1] * self.visibility_mask[np.newaxis, np.newaxis]
            new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)
            agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            new_rew = fragment.rews * agent_visible
            processed_fragments.append(TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew))
        return tuple(processed_fragments)

    # def process_preference_feedback(self, 
    #                                 fragment_pair: Tuple[TrajectoryWithRew, TrajectoryWithRew]) -> Tuple[TrajectoryWithRew, TrajectoryWithRew]:
    #     obs = np.stack([frag.obs for frag in fragment_pair])  # Shape (2, num_steps, num_features)
    #     rews = np.stack([frag.rews for frag in fragment_pair])
        
    #     masked_obs = obs[:, :, :-1] * self.visibility_mask[np.newaxis, np.newaxis, :]
    #     new_obs = np.concatenate([masked_obs, obs[:, :, -1:]], axis=-1)
    #     agent_visible = new_obs[:, :-1, 0].any(axis=(1, 2))[:, np.newaxis]  # Shape (2, 1)
    #     new_rew = rews * agent_visible
    #     processed_fragments = [
    #         TrajectoryWithRew(new_obs[i], fragment_pair[i].acts, fragment_pair[i].infos, fragment_pair[i].terminal, new_rew[i])
    #         for i in range(2)
    #     ]
    
    #     return tuple(processed_fragments)
    
    def __repr__(self):
        return f"PartialGridVisibility(grid_size={self.grid_size}, visibility_mask={self.visibility_mask}, feedback={self.feedback})"
        

# DynamicGridVisibility: This observation function masks the visibility of the agent in the gridworld environment.
# The visibility mask is a boolean matrix that is multiplied element-wise with the observation tensor. 
# The mask is applied to all channels except the last one, which is the number of carried pellets.

# The visibility mask is DYNAMIC, meaning it changes over time. The mask is defined by a pattern of camera movement, either
# hardcoded or passed as an argument. The pattern is a list of tuples, where each tuple represents the top-left corner of the camera window.
# The camera window is a n x n square centered at the top-left corner. The camera window moves according to the pattern, and wraps around the gridworld.

class DynamicGridVisibility(ObservationFunction):
    def __init__(self, 
                 env: Union[gym.Env, VecEnv],  
                 pattern: list = None, 
                 feedback: str ="scalar"):
        super().__init__()
        self.env = env
        self.grid_size = env.grid_size
        self.feedback = feedback
        
        # Define the pattern of camera movement
        if pattern is None:
            self.pattern = self.default_pattern()
        else:
            self.pattern = pattern
        print("Pattern = ", self.pattern)
        self.pattern_index = 0  # Start at the first position in the pattern

        # Build the initial visibility mask
        self.visibility_mask = self.construct_visibility_mask()

    def default_pattern(self) -> list:
        # Create a default movement pattern for the camera
        # Example for a 5x5 grid, you may adjust as needed
        positions = []
        # HARDCODED, TODO find a way to generalize this
        if self.grid_size == 3:
            positions = [(0,0), (0,1), (1,1), (1,0)]
        elif self.grid_size == 5:
            positions = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)]
        else:
            raise NotImplementedError("Default pattern not implemented for grid size other than 3x3 or 5x5")
        return positions

    def reset(self):
        self.pattern_index = 0
        self.visibility_mask = self.construct_visibility_mask()
    
    def construct_visibility_mask(self) -> np.ndarray: 
        # Build a visibility mask based on the current pattern index
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
        left_x, left_y = self.pattern[self.pattern_index]
        camera_size = self.grid_size // 2 + self.grid_size % 2
        
        # Calculate bounds of the camera window
        end_x = min(left_x + camera_size, self.grid_size)
        end_y = min(left_y + camera_size, self.grid_size)
        mask[left_x:end_x, left_y:end_y] = True
        return mask

    def update_visibility(self, 
                          t: int =None, 
                          limits=None,
                          halt: int = None) -> np.ndarray:
        # TODO (joan): Implement the halt parameter to stop the camera movement for a certain number of timesteps

        # Update the visibility mask for the next timestep

        if t is not None and limits is not None: # For the training case
            # collect t visibility masks and return a list of them
            visibility_masks = []
            self.reset()
            self.pattern_index = limits[0]
            for _ in range(t):
                self.pattern_index = (self.pattern_index) % len(self.pattern)
                visibility_masks.append(self.construct_visibility_mask())
                self.pattern_index += 1
            return np.array(visibility_masks)
        
        elif t is not None: # For the evaluation case
            visibility_masks = []
            self.reset()
            for _ in range(t):
                self.pattern_index = (self.pattern_index) % len(self.pattern)
                visibility_masks.append(self.construct_visibility_mask())
                self.pattern_index += 1
            return np.array(visibility_masks)
        
        else:
            self.pattern_index = (self.pattern_index) % len(self.pattern)
            self.visibility_mask = self.construct_visibility_mask()
            self.pattern_index += 1


    def __call__(self, 
                 fragments: Union[TrajectoryWithRew, Tuple[TrajectoryWithRew, TrajectoryWithRew]], 
                 limits=None):
        
        if self.feedback == "scalar":
            return self.process_scalar_feedback(fragments, limits)
        elif self.feedback == "preference":
            return self.process_preference_feedback(fragments, limits)

    def render_mask(self, mask):
        """ Render the visibility mask as ASCII in a more detailed and visually consistent manner. """
        print("+" + "---+" * len(mask))
        for row in mask:
            print("|", end="")
            for cell in row:
                if cell:
                    print(" # |", end="")  # Visible areas marked with '#'
                else:
                    print("   |", end="")  # Non-visible areas remain blank
            print("\n+" + "---+" * len(mask))

    def __repr__(self):
        return f"DynamicGridVisibility(\n    grid_size={self.grid_size},\n    visibility_mask=\n{self.visibility_mask},\n    feedback={self.feedback}\n)"

    def process_scalar_feedback(self, 
                                fragment: TrajectoryWithRew, 
                                limits=None) -> TrajectoryWithRew:
        
        visibility_mask = self.update_visibility(limits=limits)
        masked_obs = fragment.obs[:, :-1] * visibility_mask[np.newaxis, np.newaxis]
        new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)
        agent_visible = new_obs[:, 0, :, :].any(axis=(1, 2))
        new_rew = fragment.rews * agent_visible
        return TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, new_rew)

    def process_preference_feedback(self, 
                                    fragment_pair: Tuple[TrajectoryWithRew, TrajectoryWithRew], 
                                    limits: list[Tuple] = None) -> Tuple[TrajectoryWithRew, TrajectoryWithRew]:
        
        # limits is a list of ((start,end), (start,end)) tuples for a pair of fragments
        # Extract f1 limits from the limits list

        if limits is None:
            raise ValueError("Limits must be provided for the preference feedback. Need to know the start and end timestep of the fragments.")

        limits_f1 = limits[0]
        limits_f2 = limits[1]

        # Put vsibility mask in a dictionary
        visibility_masks_f1 = self.update_visibility(t=len(fragment_pair[0].obs), limits=limits_f1)
        visibility_masks_f2 = self.update_visibility(t=len(fragment_pair[1].obs), limits=limits_f2)
        masks = {
            0: visibility_masks_f1,
            1: visibility_masks_f2
        }
        processed_fragments = []

        for i, fragment in enumerate(fragment_pair):
            new_obs = []
            new_rews = []
            visibility_masks = masks[i]
            # visibility_masks has shape (t, grid_size, grid_size)
            # fragment.obs has shape (t, 5, grid_size, grid_size)
            # We need to apply the visibility mask to all channels except the last one
            # The last channel is the number of carried pellets, which should not be masked
            
            # Debug: find agent position in the observation
            # agent_pos = np.where(fragment.obs[:, 0, :, :] == 1)
            # time_steps = agent_pos[0]
            # x_coords = agent_pos[1]
            # y_coords = agent_pos[2]

            # print("Agent positions at time t:")
            # start = limits_f1[0]
            # j = 0
            # for t, x, y in zip(time_steps, x_coords, y_coords):
            #     print(f"t = {t} (l = {start + j}) (x={x}, y={y})")
            #     j += 1

            masked_obs = fragment.obs[:, :-1] * visibility_masks[:, np.newaxis]
            new_obs = np.concatenate([masked_obs, fragment.obs[:, -1:]], axis=1)

            # Debug: find agent position in the new observation
            # agent_pos = np.where(new_obs[:, 0, :, :] == 1)
            # time_steps = agent_pos[0]
            # x_coords = agent_pos[1]
            # y_coords = agent_pos[2]

            # print("Agent positions at time t:")
            # start = limits_f1[0]
            # j = 0
            # for t, x, y in zip(time_steps, x_coords, y_coords):
            #     print(f"t = {t} (l = {start + j}) (x={x}, y={y})")
            #     j += 1

            # apply the reward modification
            # agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            # new_rews = fragment.rews * agent_visible           
            
            # # Store the new observations and rewards in the trajectory data structure
            # new_fragment = TrajectoryWithRew(np.array(new_obs), fragment.acts, fragment.infos, fragment.terminal, np.array(new_rews))
            # processed_fragments.append(new_fragment)
            # Debug: Render the observations before and after applying the mask
            #print("Fragment {} Before Masking:".format(i+1))
            #self.env.render_rollout(TrajectoryWithRew(fragment.obs, fragment.acts, fragment.infos, fragment.terminal, fragment.rews))
            #print("Fragment {} After Masking:".format(i+1))
            #self.env.render_rollout(TrajectoryWithRew(new_obs, fragment.acts, fragment.infos, fragment.terminal, fragment.rews))

            agent_visible = new_obs[:-1, 0].any(axis=(1, 2))
            new_rews = fragment.rews * agent_visible
            new_fragment = TrajectoryWithRew(np.array(new_obs), fragment.acts, fragment.infos, fragment.terminal, np.array(new_rews))
            processed_fragments.append(new_fragment)

        return tuple(processed_fragments)

    
    def __repr__(self):
        return f"DynamicGridVisibility(pattern={self.pattern}, feedback={self.feedback})"