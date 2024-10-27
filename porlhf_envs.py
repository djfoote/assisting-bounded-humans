import numpy as np

from graph_mdp import GraphMDP, GraphObservationFunction, MatrixBeliefFunction
from partial_observability import PORLHFHumanFeedbackModel, TrajectoryWithRewAndObs


def get_hiding_env_obs_and_belief(p=0.5, r=1.0, pW=0.5, pH=0.5, pH_prime=0.5, alt_reward_fn_dict=None):
    horizon = 3  # zero-indexed
    start_states = ["S"]
    graph = {
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
    rewards = {
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
    hiding_env = GraphMDP(start_states, graph, rewards, horizon)

    observation_fn_dict = {
        "S": "o0",
        "I": "oI",
        "W": "oW",
        "L": "oL",
        "WH": "oW",
        "LH": "o0",
        "T": "o0",
    }
    observation_fn = GraphObservationFunction(hiding_env, observation_fn_dict)

    # TODO: The space delimiting is kind of brittle. It would be better to use a more robust format.
    #       Doesn't matter for these experiments.
    human_belief_function_dict = {
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
    belief_fn = MatrixBeliefFunction(hiding_env, human_belief_function_dict, observation_fn)
    return hiding_env, observation_fn, belief_fn


def get_verbose_env_obs_and_belief(p=0.5, r=1.0, pC=0.8, alt_reward_fn_dict=None):
    horizon = 3
    start_states = ["S"]
    graph = {
        "S": {
            "aD": {"L": 1.0},
            "aI": {"I": 1.0},
            "aV": {"LV": 1.0},
            "aT": {"T": 1.0},
        },
        "I": {
            "aD": {"L": 1 - p, "W": p},
            "aV": {"WV": p, "LV": 1 - p},
            "aT": {"T": 1.0},
        },
        "W": {"aT": {"T": 1.0}},
        "L": {"aT": {"T": 1.0}},
        "WV": {"aT": {"T": 1.0}},
        "LV": {"aT": {"T": 1.0}},
        "T": {"aT": {"T": 1.0}},
    }
    hiding_env_rewards = {
        "S": 0.0,
        "I": 10.0,
        "W": 5.0,
        "L": -1.0,
        "WV": 5.0 - r,
        "LV": -1.0,
        "T": 0.0,
    }
    if alt_reward_fn_dict is not None:
        raise NotImplementedError
    verbose_env = GraphMDP(start_states, graph, hiding_env_rewards, horizon)

    observation_function_dict = {
        "S": "o0",
        "I": "oI",
        "W": "o0",
        "L": "oL",
        "WV": "oW",
        "LV": "oV",
        "T": "o0",
    }
    observation_fn = GraphObservationFunction(verbose_env, observation_function_dict)

    # TODO: The space delimiting is kind of brittle. It would be better to use a more robust format.
    #       Doesn't matter for these experiments.
    human_belief_function_dict = {
        "o0 oI o0 o0": {
            "S I W T": pC,
            "S I T T": 1 - pC,
        },
    }
    belief_fn = MatrixBeliefFunction(verbose_env, human_belief_function_dict, observation_fn)
    return verbose_env, observation_fn, belief_fn


if __name__ == "__main__":
    print("Hiding environment:")
    hiding_env, hiding_obs_fn, hiding_belief_fn = get_hiding_env_obs_and_belief()
    hiding_example_ambiguous_traj = TrajectoryWithRewAndObs(
        state=np.array([hiding_env.get_state_index(state) for state in ["S", "I", "LH", "T"]]),
        obs=np.array(["o0", "oI", "o0", "o0"]),
        acts=np.array([hiding_env.get_action_index(action) for action in ["aC", "aH", "aT"]]),
        rews=np.array([0.0, 1.0, -6.0]),
        infos=None,
        terminal=True,
    )
    hiding_example_unambiguous_traj = TrajectoryWithRewAndObs(
        state=np.array([hiding_env.get_state_index(state) for state in ["S", "L", "T", "T"]]),
        obs=np.array(["o0", "oL", "o0", "o0"]),
        acts=np.array([hiding_env.get_action_index(action) for action in ["aC", "aT", "aT"]]),
        rews=np.array([0.0, -5.0, 0.0]),
        infos=None,
        terminal=True,
    )
    hiding_ambiguous_belief = hiding_belief_fn(hiding_example_ambiguous_traj)
    hiding_unambiguous_belief = hiding_belief_fn(hiding_example_unambiguous_traj)
    hiding_ambiguous_belief.pprint(hiding_env)
    hiding_unambiguous_belief.pprint(hiding_env)

    hiding_human_feedback_model = PORLHFHumanFeedbackModel(hiding_obs_fn, hiding_belief_fn, sample=False)
    hiding_fragment_pairs = [
        (hiding_example_ambiguous_traj.drop_obs(), hiding_example_ambiguous_traj.drop_obs()),
        (hiding_example_unambiguous_traj.drop_obs(), hiding_example_unambiguous_traj.drop_obs()),
        (hiding_example_ambiguous_traj.drop_obs(), hiding_example_unambiguous_traj.drop_obs()),
        (hiding_example_unambiguous_traj.drop_obs(), hiding_example_ambiguous_traj.drop_obs()),
    ]
    hiding_feedback = hiding_human_feedback_model(hiding_fragment_pairs)
    print("\nExpected: 0.5, 0.5, <large probability>, 1 - <large probability>")
    print(hiding_feedback)

    # ===============================================================================================================

    print("\nVerbose environment:")
    verbose_env, verbose_obs_fn, verbose_belief_fn = get_verbose_env_obs_and_belief()
    verbose_example_ambiguous_traj = TrajectoryWithRewAndObs(
        state=np.array([verbose_env.get_state_index(state) for state in ["S", "I", "W", "T"]]),
        obs=np.array(["o0", "oI", "o0", "o0"]),
        acts=np.array([verbose_env.get_action_index(action) for action in ["aI", "aD", "aT"]]),
        rews=np.array([0.0, 10.0, 5.0]),
        infos=None,
        terminal=True,
    )
    verbose_example_unambiguous_traj = TrajectoryWithRewAndObs(
        state=np.array([verbose_env.get_state_index(state) for state in ["S", "I", "WV", "T"]]),
        obs=np.array(["o0", "oI", "oW", "o0"]),
        acts=np.array([verbose_env.get_action_index(action) for action in ["aI", "aV", "aT"]]),
        rews=np.array([0.0, 10.0, 4.0]),
        infos=None,
        terminal=True,
    )
    verbose_ambiguous_belief = verbose_belief_fn(verbose_example_ambiguous_traj)
    verbose_unambiguous_belief = verbose_belief_fn(verbose_example_unambiguous_traj)
    verbose_ambiguous_belief.pprint(verbose_env)
    verbose_unambiguous_belief.pprint(verbose_env)

    verbose_human_feedback_model = PORLHFHumanFeedbackModel(verbose_obs_fn, verbose_belief_fn, sample=False)
    verbose_fragment_pairs = [
        (verbose_example_ambiguous_traj.drop_obs(), verbose_example_ambiguous_traj.drop_obs()),
        (verbose_example_unambiguous_traj.drop_obs(), verbose_example_unambiguous_traj.drop_obs()),
        (verbose_example_ambiguous_traj.drop_obs(), verbose_example_unambiguous_traj.drop_obs()),
        (verbose_example_unambiguous_traj.drop_obs(), verbose_example_ambiguous_traj.drop_obs()),
    ]
    verbose_feedback = verbose_human_feedback_model(verbose_fragment_pairs)
    print("\nExpected: 0.5, 0.5, 0.5, 0.5 (pC=0.8 is the break-even point between these two trajectories)")
    print(verbose_feedback)
