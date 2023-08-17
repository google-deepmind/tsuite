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
"""Tests the causality inference of the agent."""

from collections.abc import Sequence
import random

import dm_env
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the causality inference of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of a specific observation, a second
  observation is provided which is merely correlated with reward.

  Every episode in this test case is of length 1. The observation in an episode
  can either contain a signal or not. If the observation contains a signal, the
  agent is expected to output the "high" action, otherwise the agent is expected
  to output the "low" action to get reward.

  This test ensures that the agent infers the correct causal structure.
  """

  def __init__(
      self,
      causal_observation_name: str,
      correlated_observation_name: str,
      correlation_percentage: str,
      **kwargs):
    kwargs["default_observation_name"] = causal_observation_name
    self._causal_observation_name = causal_observation_name
    self._correlated_observation_name = correlated_observation_name
    self._correlation_percentage = int(correlation_percentage)
    super().__init__(**kwargs)

  def reset(self) -> dm_env.TimeStep:
    # The episode counter is not yet increased, so we check for == 1 here
    # instead of == 0.
    signal_obs = []
    if (self.episode_counter % 2) == 1:
      signal_obs.append(self._causal_observation_name)
      if (random.random() * 100) < self._correlation_percentage:
        signal_obs.append(self._correlated_observation_name)
    observation = tree.map_structure(
        base.make_signal_injector_visitor_fn(signal_obs),
        self._observation_spec)
    return super().base_reset(observation=observation)

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    internal_action = self.map_external_to_internal_action(action)
    return super().base_step(
        success=internal_action == self.best_next_internal_action(),
        observation=self.get_observation(),
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
  names = [spec.name for spec in tree.flatten(observation_spec)]
  tasks = []
  for n1 in names:
    for n2 in names:
      if n1 != n2:
        for percentage in [90, 99]:
          tasks.append(f"causal@{n1}@{n2}@{percentage}")
  return tasks
