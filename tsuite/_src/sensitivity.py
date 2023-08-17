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
"""Tests the sensitivity of the agent to small numerical observations."""

from collections.abc import Sequence
import itertools

import dm_env
import numpy as np
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the visual capabilities of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of a numerical observation.
  The signal-size of the numerical observation can be reduced in powers of ten.
  A sensitivity of n means that the agent has to distinguish 0 and 10^n.
  A sensitivity of 1 means that the agent has to distinguish 0 and 10.
  A sensitivity of 0 means that the agent has to distinguish 0 and 1.
  A sensitivity of -1 means that the agent has to distinguish 0 and 0.1.
  A sensitivity of -n means that the agent has to distinguish 0 and 10^-n.
  """

  def __init__(self, observation_name: str, sensitivity: str, **kwargs):
    kwargs["default_observation_name"] = observation_name
    super().__init__(**kwargs)
    self._factor = 10**int(sensitivity)

    def high_visitor(path, node):
      if path == self._default_observation_path:
        return np.full_like(node, self._factor)
      return node

    def low_visitor(path, node):
      if path == self._default_observation_path:
        return np.zeros_like(node)
      return node

    self._numeric_observations = [
        tree.map_structure_with_path(low_visitor, self.get_observation()),
        tree.map_structure_with_path(high_visitor, self.get_observation())
    ]

  def reset(self) -> dm_env.TimeStep:
    return super().base_reset(
        observation=self._numeric_observations[self.episode_counter % 2])

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    internal_action = self.map_external_to_internal_action(action)
    return super().base_step(
        success=internal_action == self.best_next_internal_action(),
        terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    signal = (self.episode_counter % 2) == 0
    return base.InternalAction.HIGH if signal else base.InternalAction.LOW


def list_test_tasks(observation_spec: base.SpecsTree) -> Sequence[str]:
  """Returns available test-tasks of TestCase.

  Args:
    observation_spec: defines the observations consumed by the agent.
  """
  names = []
  for node in tree.flatten(observation_spec):
    if np.issubdtype(node.dtype, np.number):
      names.append(node.name)
  return [f"sensitivity@{name}@{n}"
          for name, n in itertools.product(names, [-1, 0, 1])]


