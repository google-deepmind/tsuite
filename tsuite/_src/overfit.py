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
"""Tests the ability of the agent to overfit to a sequence."""

from collections.abc import Sequence

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the capability of the agent to overfit to a sequence.

  The agent only receives rewards if it learns to output certain sequence of
  high and low values for the default action:
    (1xL, 2xH, 1xL).

  In total there are 4 actions, hence it is likely that the agent can discover
  the solution on its own.
  """

  def __init__(self, **kwargs):
    """Initializes a new TestCase.

    Args:
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    self._secret = [base.InternalAction.LOW,
                    base.InternalAction.HIGH, base.InternalAction.HIGH,
                    base.InternalAction.LOW]
    self._sequence = []

  def reset(self) -> dm_env.TimeStep:
    self._sequence = []
    return super().base_reset()

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    self._sequence.append(self.map_external_to_internal_action(action))
    if len(self._sequence) < len(self._secret):
      return super().base_step()
    return super().base_step(
        success=self._sequence == self._secret, terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    n = len(self._sequence)
    if n >= len(self._secret):
      return self._secret[0]
    return self._secret[n]


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["overfit"]
