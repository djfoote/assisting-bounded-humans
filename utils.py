from stealing_gridworld import StealingGridworld
import gym
import functools
from typing import Callable, List, Mapping, Optional, Sequence, Union, Any
import numpy as np
#from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from imitation.data.types import TrajectoryWithRew




def make_vec_env(
    env_name: str,
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
    is_custom: bool = False,  # Add a flag to indicate if the environment is a custom one
) -> VecEnv:
    """Makes a vectorized environment for both Gym and custom environments."""

    env_make_kwargs = env_make_kwargs or {}

    if not is_custom:
        # Standard Gym environment initialization
        tmp_env = gym.make(env_name)
        tmp_env.close()
        spec = tmp_env.spec
    else:
        # Custom environment handling
        spec = None

    def make_env(i: int, this_seed: int) -> gym.Env:
        if not is_custom:
            env = gym.make(env_name, **env_make_kwargs)
            if max_episode_steps is not None:
                from gym.wrappers import TimeLimit
                env = TimeLimit(env, max_episode_steps=max_episode_steps)
        else:
            # Here you need to modify how your custom environment is created,
            # ensuring it accepts necessary initialization parameters.
            env = StealingGridworld(**env_make_kwargs)

        # Seed each environment with a different, non-sequential seed for diversity
        env.seed = this_seed

        # Apply additional wrappers if any
        if post_wrappers:
            env = post_wrappers(env)

        #return FlattenObservation(env)
        return env

    # Generate unique seeds for each environment
    env_seeds = make_seeds(rng, n_envs)
    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
    ]

    # Choose between parallel and sequential multi-environment handling
    if parallel:
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)

def make_seeds(
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> Union[Sequence[int], int]:
    """Generate n random seeds from a random state."""
    seeds = rng.integers(0, (1 << 31) - 1, (n if n is not None else 1,))
    seeds_list: List[int] = seeds.tolist()
    if n is None:
        return seeds_list[0]
    else:
        return seeds_list
    

from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import numpy as np
import gym
import warnings
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, is_vecenv_wrapped
from imitation.data.types import TrajectoryWithRew  # Import the TrajectoryWithRew class

from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import numpy as np
import gym
import warnings
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, is_vecenv_wrapped
from imitation.data.types import TrajectoryWithRew  # Import the TrajectoryWithRew class

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    return_trajectories: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]], Tuple[List[float], List[int], List[TrajectoryWithRew]]]:
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    trajectories = [None] * n_envs

    episode_counts = np.zeros(n_envs, dtype=int)
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype=int)
    # observations = env.reset()
    
    
    states = None
    episode_starts = np.ones(n_envs, dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        new_observations, rewards, dones, infos = env.step(actions)

        for i in range(n_envs):
            if trajectories[i] is None:
                trajectories[i] = {
                    'obs': [],
                    'acts': [],
                    'rews': [],
                    'terminal': False,
                    'infos': None
                }

            trajectories[i]['obs'].append(observations[i])
            trajectories[i]['acts'].append(actions[i])
            trajectories[i]['rews'].append(rewards[i])

            current_rewards[i] += rewards[i]
            current_lengths[i] += 1

            if dones[i]:
                trajectories[i]['terminal'] = dones[i]
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                episode_counts[i] += 1
                current_rewards[i] = 0
                current_lengths[i] = 0
                # Convert list data to TrajectoryWithRew
                completed_trajectory = TrajectoryWithRew(
                    obs=np.array(trajectories[i]['obs'], dtype=np.int16),
                    acts=np.array(trajectories[i]['acts'], dtype=np.int16),
                    rews=np.array(trajectories[i]['rews'], dtype=float),
                    terminal=trajectories[i]['terminal'],
                    infos=trajectories[i]['infos']
                )
                trajectories[i] = completed_trajectory
                trajectories[i] = None

        observations = new_observations
        if render:
            env.render()

    # Filter out any None placeholders left from incomplete episodes
    trajectories = [traj for traj in trajectories if traj is not None]

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"

    if return_episode_rewards and not return_trajectories:
        return episode_rewards, episode_lengths
    elif return_trajectories:
        return episode_rewards, episode_lengths, trajectories
    return mean_reward, std_reward