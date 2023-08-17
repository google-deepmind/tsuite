# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests the reaction of the agent to user-provided inputs."""

from collections.abc import Mapping, Sequence
import pickle
from typing import TypeVar

import dm_env
import numpy as np
import tree

from tsuite._src import base

internal_open = open


ArrayT = TypeVar("ArrayT", bound=base.ArrayTree)


def tree_update(lhs: ArrayT, rhs: base.ArrayTree) -> ArrayT:
  """Returns updated array trees.
  
  The lhs provides the default values, which are potentially overwritten by
  the rhs. The right hand side does not need to be of the exact same type as
  the lhs, as long as the path structure matches.
  e.g. lhs = dm_env.Timestep(observation, reward, step_type, discount) can be
  updated using a dictionary dict(reward=2).
  
  Args:
    lhs: left hand side that provides defaults.
    rhs: right hand side that provides overrides.
  """
  flat_lhs = tree.flatten_with_path(lhs)
  flat_rhs = tree.flatten_with_path(rhs)
  map_rhs = dict(flat_rhs)
  updated_lhs = [map_rhs.get(path, value) for path, value in flat_lhs]
  return tree.unflatten_as(lhs, updated_lhs)


def tree_intersection_compare(lhs: base.ArrayTree, rhs: base.ArrayTree) -> bool:
  """Returns whether the two given arrays trees are equal.
  
  Only elements present in both structures need to be equal.
  In particular tree_intersection_compare(AnyTree, {}) would always be True.
  
  Args:
    lhs: left hand side.
    rhs: right hand side.
  """
  flat_lhs = tree.flatten_with_path(lhs)
  flat_rhs = tree.flatten_with_path(rhs)
  map_rhs = dict(flat_rhs)
  for path, lhs_value in flat_lhs:
    if path in map_rhs:
      rhs_value = map_rhs[path]
      if np.any(lhs_value != rhs_value):
        return False
  return True


class TestCase(base.TestCase):
  """Tests the reaction of the agent to user-provided inputs.

  The agent receives a reward if its able to output a high / low value
  for a given user-provided input.

  This test enables users to create custom tsuite tests on the fly, by
  providing observations and the expected action.

  Custom user input format
  ========================

  The user provides the custom input as a pickle file. The pickle file must
  contain a Sequence of episodes.

  Episode:
  -------
  Each episode must consists of Mapping with three optional entries "timestep",
  "internal_action" or "external_action".
  If present the value of the "timestep" entry must be an ArrayTree that can be
  used to update the default timestep returned by tsuite.
  If present the value of "internal_action" or "external_action" is compared to
  the action returned by the agent for this timestep, and will determine the
  reward associated with the next default timestep returned by tsuite.
  "internal_action" has to be given in the tsuite internal format
    where 0 refers to a low-action, 1 to a noop, and 2 to a high-action.
  "external_action" has to be given in the action_spec of the environment,
    the external_action is compared using `tree_intersection_compare`.

  TSuite default timesteps:
  ------------------------
  For the first timestep in the episode tsuite returns:
    dm_env.TimeStep(observation=zero_obs_according_to_spec,
                    step_type=first
                    reward=None,
                    discount=None)

  For the last timestep in the episode tsuite returns:
    dm_env.TimeStep(observation=zero_obs_according_to_spec,
                    step_type=last
                    # determined by "internal_action" or "external_action"
                    reward=reward,
                    discount=0.0)

  For all other timesteps in the episode tsuite returns:
    dm_env.TimeStep(observation=zero_obs_according_to_spec,
                    step_type=mid
                    # determined by "internal_action" or "external_action"
                    reward=reward,
                    discount=1.0)

  Example: Matching colors
  ------------------------

  Given an observation spec: {'rgb': Array(shape=(96, 72, 3))},
  and an action spec: {'color': DiscreteArray(num_values=3)}
  A valid custom test that checks if an agent can react with the correct color.

  # Sequence of episodes
  [
      # First episode "red"
      [
          {
              'timestep': dict(observation={'rgb': red_array}),
              'external_action': {'color': np.array(0)}
          },
          {}, # Use default tsuite timestep for last-timestep.
      ],
      # Second episode "green"
      [
          {
              'timestep': dict(observation={'rgb': green_array}),
              'external_action': {'color': np.array(1)}
          },
          {}, # Use default tsuite timestep for last-timestep.
      ],
      # Third episode "blue"
      [
          {
              'timestep': dict(observation={'rgb': blue_array}),
              'external_action': {'color': np.array(2)}
          },
          {}, # Use default tsuite timestep for last-timestep.
      ]
  ]
  """

  def __init__(self, path: str, **kwargs):
    """Initializes a new TestCase.

    Args:
      path: path to the user provided file in pickle format.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    with internal_open(path, "rb") as f:
      self._episodes = pickle.load(f)
    if not isinstance(self._episodes, Sequence):
      raise ValueError(f"Provided episodes are malformed: {self._episodes}")
    for episode in self._episodes:
      if not isinstance(episode, Sequence):
        raise ValueError(f"Provided episode is malformed: {episode}")
      if len(episode) < 2:
        raise ValueError(f"Provided episode is too short: {len(episode)}")
      for step in episode:
        if not isinstance(step, Mapping):
          raise ValueError(f"Provided episode is malformed: {episode}")
    self._episode = self._episodes[0]

  def reset(self) -> dm_env.TimeStep:
    self._episode = self._episodes[
        self.episode_counter % len(self._episodes)]
    timestep = self.base_reset()
    if "timestep" in self._episode[0]:
      timestep = tree_update(timestep, self._episode[0]["timestep"])
    return timestep

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    success = False
    last_step = self._episode[self.step_counter]
    if "internal_action" in last_step:
      internal_action = self.map_external_to_internal_action(action)
      success = internal_action == last_step["internal_action"]
    elif "external_action" in last_step:
      success = tree_intersection_compare(action, last_step["external_action"])
    timestep = super().base_step(
        success=success,
        # Step counter is updated in base_step.
        terminate=len(self._episode) == self.step_counter + 2)
    if "timestep" in self._episode[self.step_counter]:
      timestep = tree_update(
          timestep, self._episode[self.step_counter]["timestep"])
    return timestep

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    if "internal_action" in self._episode[self.step_counter]:
      return self._episode[self.step_counter]["internal_action"]
    elif "external_action" in self._episode[self.step_counter]:
      return self.map_external_to_internal_action(
          self._episode[self.step_counter]["external_action"])
    return base.InternalAction.NOOP


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  # There are no default custom test-cases. If one would exists it would look
  # like this:
  # return ["custom@/home/user/custom.pickle"]
  return []
