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
"""Simulates an environment which is not thread safe."""

from collections.abc import Sequence
import threading
import time

import dm_env

from tsuite._src import base

# Global lock.
_lock = threading.Lock()


class TestCase(base.TestCase):
  """Simulates an environment which is not thread safe.

  This TestCase simulates a broken environment; it is reasonable for an agent
  to crash with a meaningful error message.

  This test always returns a reward regardless of the agent input.
  """

  def reset(self) -> dm_env.TimeStep:
    if _lock.locked():
      raise RuntimeError("Encountered thread safety issue!")
    with _lock:
      # Sleeps for 100ms to increase chance of collisions between threads.
      time.sleep(0.1)
      return super().base_reset()

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    if _lock.locked():
      raise RuntimeError("Encountered thread safety issue!")
    with _lock:
      # Sleeps for 100ms to increase chance of collisions between threads.
      time.sleep(0.1)
      return super().base_step(success=True, terminate=True)


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["thread_safety"]
