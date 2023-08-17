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
"""Tests if agent works with zero discount."""

from collections.abc import Sequence


import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if agent works with zero discount."""

  def reset(self) -> dm_env.TimeStep:
    return super().base_reset(
        observation=self.get_observation(signal=(self.episode_counter % 2) == 0)
        )

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    internal_action = self.map_external_to_internal_action(action)
    timestep = super().base_step(
        success=internal_action == self.best_next_internal_action(),
        terminate=self.step_counter == 1)
    timestep = timestep._replace(discount=0.0)
    return timestep

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    signal = (self.episode_counter % 2) == 0
    return base.InternalAction.HIGH if signal else base.InternalAction.LOW


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["zero_discount"]
