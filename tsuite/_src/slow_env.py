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
"""Tests the ability of the agent to deal with a slow environment."""

from collections.abc import Sequence
import time

import dm_env

from tsuite._src import base
from tsuite._src import overfit


class TestCase(overfit.TestCase):
  """Tests the capability of the agent to learn from a slow environment.

  This is the same as the overfit test, except for an additional configurable
  sleep parameter that simulates a long step time of the underlying environment.
  """

  def __init__(self, delay_in_ms: str, **kwargs):
    """Initializes a new TestCase.

    Args:
      delay_in_ms: delay in milliseconds.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    self._delay_in_ms = int(delay_in_ms)
    if self._delay_in_ms <= 0:
      raise ValueError("Delay must be positive.")

  def reset(self) -> dm_env.TimeStep:
    time.sleep(self._delay_in_ms / 1000.0)
    return super().reset()

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    time.sleep(self._delay_in_ms / 1000.0)
    return super().step(action)


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["slow_env@500"]
