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
"""Simulates an environment which crashing with a certain probability."""

from collections.abc import Sequence
import random

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Simulates an environment which crashes with a certain probability.

  This TestCase simulates a broken environment; it is reasonable for an agent
  to crash with a meaningful error message.

  This test always returns a reward regardless of the agent input.
  """

  def __init__(self, crash_probability_in_percent: str, **kwargs):
    """Initializes a new TestCase.

    Args:
      crash_probability_in_percent: crash probability in percent.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    self._crash_probability_in_percent = int(crash_probability_in_percent)
    if self._crash_probability_in_percent < 0:
      raise ValueError("crash_probability_in_percent must be positive.")
    if self._crash_probability_in_percent > 100:
      raise ValueError("crash_probability_in_percent must be <= 100.")

  def reset(self) -> dm_env.TimeStep:
    self._maybe_raise_intentional_error()
    return super().base_reset()

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    self._maybe_raise_intentional_error()
    return super().base_step(success=True, terminate=True)

  def _maybe_raise_intentional_error(self):
    if random.random() * 100 < self._crash_probability_in_percent:
      raise RuntimeError("This is an intentional error for testing purposes.")


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["crashing_env@1"]
