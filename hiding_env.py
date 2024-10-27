import numpy as np

from graph_mdp import GraphMDP, GraphObservationFunction, MatrixBeliefFunction
from partial_observability import PORLHFHumanFeedbackModel, TrajectoryWithRewAndObs


def get_hiding_env_obs_and_belief(p=0.5, r=1.0, pW=0.5, pH=0.5, pH_prime=0.5, alt_reward_fn_dict=None):
    p = 0.5
    r = 1.0
    hiding_env_horizon = 3
    hiding_env_start_states = ["S"]
    hiding_env_graph = {
        "S": {
            "aC": {"L": 1.0},
            "aI": {"I": 1.0},
            "aH": {"LH": 1.0},
            "aT": {"T": 1.0},
        },
        "I": {
            "aC": {"L": 1 - p, "W": p},
            "aH": {"WH": p, "LH": 1 - p},
            "aT": {"T": 1.0},
        },
        "W": {"aT": {"T": 1.0}},
        "L": {"aT": {"T": 1.0}},
        "WH": {"aT": {"T": 1.0}},
        "LH": {"aT": {"T": 1.0}},
        "T": {"aT": {"T": 1.0}},
    }
    hiding_env_rewards = {
        "S": 0.0,
        "I": 1.0,
        "W": 10.0,
        "L": -5.0,
        "WH": 10.0,
        "LH": -5.0 - r,
        "T": 0.0,
    }
    if alt_reward_fn_dict is not None:
        raise NotImplementedError
    hiding_env = GraphMDP(hiding_env_start_states, hiding_env_graph, hiding_env_rewards, hiding_env_horizon)

    hiding_env_observation_function_dict = {
        "S": "o0",
        "I": "oI",
        "W": "oW",
        "L": "oL",
        "WH": "oW",
        "LH": "o0",
        "T": "o0",
    }
    observation_fn = GraphObservationFunction(hiding_env, hiding_env_observation_function_dict)

    # TODO: The space delimiting is kind of brittle. It would be better to use a more robust format.
    #       Doesn't matter for these experiments.
    hiding_env_human_belief_function_dict = {
        "o0 oI o0 o0": {
            "S I LH T": pH,
            "S I T T": 1 - pH,
        },
        "o0 o0 o0 o0": {
            "S LH T T": pH_prime,
            "S T T T": 1 - pH_prime,
        },
        "o0 oI oW o0": {
            "S I W T": pW,
            "S I WH T": 1 - pW,
        },
    }
    belief_fn = MatrixBeliefFunction(
        hiding_env,
        hiding_env_human_belief_function_dict,
        observation_fn,
    )
    return hiding_env, observation_fn, belief_fn


if __name__ == "__main__":
    hiding_env, observation_fn, belief_fn = get_hiding_env_obs_and_belief()
    example_traj = TrajectoryWithRewAndObs(
        state=np.array(["S", "I", "LH", "T"]),
        obs=np.array(["o0", "oI", "o0", "o0"]),
        acts=np.array(["aC", "aH", "aT"]),
        rews=np.array([0.0, 1.0, 0.0]),
        infos=None,
        terminal=True,
    )
    example_unambiguous_traj = TrajectoryWithRewAndObs(
        state=np.array(["S", "L", "T", "T"]),
        obs=np.array(["o0", "oL", "o0", "o0"]),
        acts=np.array(["aC", "aT", "aT"]),
        rews=np.array([0.0, -5.0, 0.0]),
        infos=None,
        terminal=True,
    )
    ambiguous_belief = belief_fn(example_traj)
    unambiguous_belief = belief_fn(example_unambiguous_traj)
    print(ambiguous_belief)
    print(unambiguous_belief)

    human_feedback_model = PORLHFHumanFeedbackModel(observation_fn, belief_fn, sample=False)
    fragment_pairs = [
        (example_traj.drop_obs(), example_traj.drop_obs()),
        (example_unambiguous_traj.drop_obs(), example_unambiguous_traj.drop_obs()),
        (example_traj.drop_obs(), example_unambiguous_traj.drop_obs()),
        (example_unambiguous_traj.drop_obs(), example_traj.drop_obs()),
    ]
    feedback = human_feedback_model(fragment_pairs)
    print("\nExpected: 0.5, 0.5, <large probability>, 1 - <large probability>")
    print(feedback)
