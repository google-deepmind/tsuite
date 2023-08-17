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
"""Tests the observation space of the agent."""

from collections.abc import Sequence

import dm_env
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the observation space of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of a specific observation.

  Every episode in this test case is of length 1. The observation in an episode
  can either contain a signal or not. If the observation contains a signal, the
  agent is expected to output the "high" action, otherwise the agent is expected
  to output the "low" action to get reward.

  This test ensures that the agent is sensitive to its observation space.
  """

  def __init__(self, observation_name: str, **kwargs):
    kwargs["default_observation_name"] = observation_name
    super().__init__(**kwargs)

  def reset(self) -> dm_env.TimeStep:
    # The episode counter is not yet increased, so we check for == 1 here
    # instead of == 0.
    signal = (self.episode_counter % 2) == 1
    return super().base_reset(observation=self.get_observation(signal))

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
  return [f"observation_space@{name}" for name in names]
