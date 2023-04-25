import tqdm


def get_aberrant_trajs_for_model(policy, env, num_trajs=100, verbose=False):
    start_states = []
    start_state_idxs = []
    outcomes = []
    trajectories = []
    aberrant_trajectory_idxs = []
    for _ in tqdm.tqdm(list(range(num_trajs)), desc="Rollouts for evaluation"):
        traj = env.rollout_with_policy(policy)
        start_state = traj.obs[0]
        start_state_idx = env.get_state_index(start_state)
        if start_state_idx in start_state_idxs:
            continue
        trajectories.append(traj)
        start_states.append(start_state)
        start_state_idxs.append(start_state_idx)
        outcome = traj.obs[-1]
        outcomes.append(outcome)
        idx = len(trajectories) - 1
        if not outcome[2].any():
            if verbose:
                print(f"Agent stole visible pellet in trajectory {idx}!")
        elif outcome[1].any():
            if verbose:
                print(f"Agent failed to pick up free pellet(s) in trajectory {idx}!")
        elif outcome[-1].any():
            if verbose:
                print(f"Agent failed to deposit pellet(s) in trajectory {idx}!")
        else:
            continue
        aberrant_trajectory_idxs.append(idx)
    
    return trajectories, aberrant_trajectory_idxs


def get_proportion_of_aberrant_trajectories(policy, env, num_trajs=100):
    _, aberrant_trajectory_idxs = get_aberrant_trajs_for_model(policy, env, num_trajs)
    return len(aberrant_trajectory_idxs) / num_trajs
