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
"""Checks for cross contamination between episodes."""

from collections.abc import Sequence

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if episodes are cross contaminated.

  This test can only be solved by leaking information between episodes. If an
  agent passes this test, it indicates a problem with the resetting of its
  memory. In particular, the last observation leaks into the next episode.
  """

  def expected_reward_to_pass_test(self):
    # We expect the agent to NOT learn to solve this task!
    return (base.SUCCESS + base.FAIL) / 2

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    expected = self.best_next_internal_action()
    next_signal = (self.episode_counter + 1) % 2 == 0
    # Observation in the last time-step, leaks the secret of the next episode!
    return super().base_step(
        success=self.map_external_to_internal_action(action) == expected,
        observation=self.get_observation(next_signal),
        terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    if (self.episode_counter % 2) == 0:
      return base.InternalAction.HIGH
    else:
      return base.InternalAction.LOW


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["cross_contamination"]
