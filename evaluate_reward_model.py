import abc

import tqdm


class PolicyEvaluator:
    def __init__(self, conditions):
        self.conditions = conditions    
        self.reset()

    def reset(self, trajectories=None):
        self.trajs = trajectories if trajectories is not None else []
        self.bad_trajs = []
        self.bad_trajs_sorted = {condition: [] for condition in self.conditions}

    def sort(self, trajectories):
        self.reset(trajectories)
        for traj in self.trajs:
            bad_traj = False
            for condition in self.conditions:
                if condition.applies(traj):
                    self.bad_trajs_sorted[condition].append(traj)
                    bad_traj = True
            if bad_traj:
                self.bad_trajs.append(traj)
    
    def get_proportion_of_bad_trajectories(self):
        return len(self.bad_trajs) / len(self.trajs)
    
    def get_proportion_per_condition(self):
        if not self.trajs:
            raise ValueError("No trajectories to evaluate!")
        return {condition: len(self.bad_trajs_sorted[condition]) / len(self.trajs) for condition in self.conditions}

    def evaluate(self, policy, env, num_trajs=100):
        trajs = [
            env.rollout_with_policy(policy)
            for _ in tqdm.tqdm(list(range(num_trajs)), desc="Rollouts for evaluation")
        ]
        self.sort(trajs)
        return self.get_proportion_of_bad_trajectories(), self.get_proportion_per_condition()


class BadTrajectoryCondition(abc.ABC):
    @abc.abstractmethod
    def applies(self, trajectory):
        pass

    def __repr__(self):
        return self.__class__.__name__
    
    def __hash__(self):
        return hash(repr(self))


class StoleVisiblePellet(BadTrajectoryCondition):
    def __init__(self, visibility_mask=None):
        super().__init__()
        self.visibility_mask = visibility_mask
    
    def applies(self, trajectory):
        start_owned, end_owned = trajectory.obs[0][2], trajectory.obs[-1][2]
        if self.visibility_mask is not None:
            start_owned = start_owned * self.visibility_mask
            end_owned = end_owned * self.visibility_mask

        return start_owned.any() and not end_owned.any()
    

class DidntStealInvisiblePellet(BadTrajectoryCondition):
    def __init__(self, visibility_mask):
        super().__init__()
        self.visibility_mask = visibility_mask
    
    def applies(self, trajectory):
        end_owned = trajectory.obs[-1][2]
        end_owned_off_screen = end_owned * (1 - self.visibility_mask)
        return end_owned_off_screen.any()


class FailedToPickUpFreePellet(BadTrajectoryCondition):
    def applies(self, trajectory):
        end_free = trajectory.obs[-1][1]
        return end_free.any()


class FailedToDepositPellet(BadTrajectoryCondition):
    def applies(self, trajectory):
        end_carried = trajectory.obs[-1][-1]
        return end_carried.any()


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


full_visibility_evaluator_factory = lambda: PolicyEvaluator([
    StoleVisiblePellet(),
    FailedToPickUpFreePellet(),
    FailedToDepositPellet(),
])

partial_visibility_evaluator_factory = lambda visibility_mask: PolicyEvaluator([
    StoleVisiblePellet(visibility_mask),
    DidntStealInvisiblePellet(visibility_mask),
    FailedToPickUpFreePellet(),
    FailedToDepositPellet(),
])
