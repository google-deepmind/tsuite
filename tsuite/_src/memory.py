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
"""Tests the memory of the agent."""

from collections.abc import Sequence

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the memory capabilities of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of a specific observation. The agent is
  required to output the action with a delay of `n_steps`.

  This test ensures that the agent is able to memorize information during the
  episode.
  """

  def __init__(self, n_steps: str, **kwargs):
    """Initializes a new MemoryTestCase.

    Args:
      n_steps: delay between the observation of the signal and the action.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    self._n_steps = int(n_steps)

  def reset(self) -> dm_env.TimeStep:
    # The episode counter is not yet increased, so we check for == 1 here
    # instead of == 0.
    signal = (self.episode_counter % 2) == 1
    return super().base_reset(observation=self.get_observation(signal))

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    # Waits for n_steps before giving a reward for the correct action.
    if self.step_counter < self._n_steps:
      return super().base_step()
    internal_action = self.map_external_to_internal_action(action)
    return super().base_step(
        success=internal_action == self.best_next_internal_action(),
        observation=self.get_observation(),
        terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    if self.step_counter < self._n_steps:
      return base.InternalAction.NOOP
    signal = (self.episode_counter % 2) == 0
    return base.InternalAction.HIGH if signal else base.InternalAction.LOW


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return [f"memory@{n}" for n in range(10)]
