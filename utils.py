import gym
import os
import functools
from typing import Callable, List, Mapping, Optional, Sequence, Union, Any
import numpy as np
#from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from gymnasium.wrappers import FlattenObservation


from stealing_gridworld import StealingGridworld

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

        # Optionally wrap each environment in a Monitor for logging
        log_path = None
        # if log_dir is not None:
        #     log_subdir = os.path.join(log_dir, "monitor")
        #     os.makedirs(log_subdir, exist_ok=True)
        #     log_path = os.path.join(log_subdir, f"mon{i:03d}")
        #env = Monitor(env, log_path)

        # Apply additional wrappers if any
        if post_wrappers:
            #for wrapper in post_wrappers:
            #    env = wrapper(env, i)
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