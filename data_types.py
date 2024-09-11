from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from deterministic_mdp import DeterministicMDP


ActionType = TypeVar("ActionType")


class ActionSpace(tuple, Generic[ActionType]):
    # Functionally identical to a tuple; exists for type checking purposes.
    pass


@dataclass(frozen=True)
class Trajectory:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    infos: list = field(default_factory=list)

    env: "DeterministicMDP" = None

    @property
    def state(self):
        return self.states[-1] if self.states else None

    @property
    def action(self):
        return self.actions[-1] if self.actions else None

    @property
    def info(self):
        return self.infos[-1] if self.infos else None

    @property
    def done(self):
        return self.env.is_terminal(self.state) if self.env is not None else False

    @cached_property
    def rewards(self):
        # NOTE: ignoring last state as it will be the state we transition to after the last action
        # (there will be one more state relative to the number of actions)
        return [self.env.reward_function(s, a) for s, a in zip(self.states[:-1], self.actions)]

    @property
    def reward(self):
        return self.rewards[-1]

    @cached_property
    def visited(self):
        return set(deep_hash(state) for state in self.states)

    def contains(self, state):
        return deep_hash(state) in self.visited

    def step(self, action):
        return self.env.step(self, action)

    def render(self):
        return self.env.render(self)

    def append_copy(self, state=None, action=None, info=None):
        info = info if info is not None else {}

        return Trajectory(
            states=self.states + [state],
            actions=self.actions + [action],
            infos=self.infos + [info],
            env=self.env,
        )

    def edit_copy(
        self,
        state=None,
        action=None,
        info=None,
        states=None,
        actions=None,
        infos=None,
        env=None,
    ):
        assert state is None or states is None
        assert action is None or actions is None
        assert info is None or infos is None

        states = states if states is not None else self.states[:-1] + [state] if state is not None else self.states
        acts = actions if actions is not None else self.actions[:-1] + [action] if action is not None else self.actions
        infos = infos if infos is not None else self.infos[:-1] + [info] if info is not None else self.infos
        env = env if env is not None else self.env

        return Trajectory(states, acts, infos, env)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        counterfactual_rewards = None
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.states))
            assert step == 1
            states = self.states[start:stop]
            actions = self.actions[start : stop - 1]
        else:
            states = [self.states[index]]
            actions = [self.actions[index]] if index < len(self.actions) else []
        return Trajectory(
            states=states,
            actions=actions,
            env=self.env,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def iter_decisions(self):
        for i in range(len(self) - 1):
            yield self[: i + 1], self.actions[i]

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False

        same_states = deep_hash(self.states) == deep_hash(other.states)
        same_actions = deep_hash(self.actions) == deep_hash(other.actions)
        same_env = self.env == other.env  # For now, check if they are the same object. This is probably enough.
        # TODO: Should we compare infos? Probably not, since they are not used in the MDP.

        return same_states and same_actions and same_env

    def __hash__(self):
        # This differs in behavior from __eq__ if you hash trajectories from different environments. I can't think of a
        # good reason to do that, but it's worth noting.
        return hash((deep_hash(self.states), deep_hash(self.actions)))


def deep_hash(obj):
    if isinstance(obj, np.ndarray) or isinstance(obj, tuple) or isinstance(obj, list):
        return tuple(deep_hash(x) for x in obj)
    elif hasattr(obj, "__hash__"):
        return obj.__hash__()
    else:
        raise ValueError(f"Object of type {type(obj)} is not hashable.")
