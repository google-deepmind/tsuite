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
"""Tests if agent uses its discount correctly.

The overall undiscounted return of the agent is 1.0 if the agent uses its
discount rate correctly, otherwise it will be larger or smaller than 1.0.
"""

from collections.abc import Sequence

import dm_env
import numpy as np

from tsuite._src import base


N_OPTIMAL_STEPS = 9
# Approximation of partial haromonic sum: H_n = sum_{k=1}^{n} 1/k
_HARMONIC_FACTOR = (
    np.log(N_OPTIMAL_STEPS-1) + np.euler_gamma + 1 / (2 * (N_OPTIMAL_STEPS-1)))


class TestCase(base.TestCase):
  """Tests if agent takes its discount correctly into account."""

  def __init__(self, target_discount: str, **kwargs):
    """Initializes a new ActionSpaceTestCase.

    Args:
      target_discount: discount-rate of the agent.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    target_discount = float(target_discount)
    # The agent will receive a linearly decreasing rewards alpha/step_count
    # and a constant terminal reward. Alpha is chosen such that the discounted
    # reward is maximal if the agent terminates the episode after
    # _N_OPTIMAL_STEPS steps, and the overall return will be 1.0 in that case.
    self._alpha = (1 - target_discount) * N_OPTIMAL_STEPS
    self._terminal_reward = 1 / (1 + self._alpha * _HARMONIC_FACTOR)
    super().__init__(**kwargs)

  def _get_intermediate_reward(self) -> float:
    return self._terminal_reward * self._alpha / self.step_counter

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    self.step_counter += 1
    internal_action = self.map_external_to_internal_action(action)
    observation = self.get_observation()
    if internal_action == base.InternalAction.HIGH:
      return dm_env.termination(
          reward=self._terminal_reward, observation=observation)
    else:
      return dm_env.transition(
          reward=self._get_intermediate_reward(), observation=observation)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    if (self.step_counter + 1) < N_OPTIMAL_STEPS:
      return base.InternalAction.LOW
    return base.InternalAction.HIGH


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["discount@0.99"]
