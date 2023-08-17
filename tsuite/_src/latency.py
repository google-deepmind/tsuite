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
"""Tests the latency of the agent-runloop."""

from collections.abc import Sequence
import time

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if the latency is below a given threshold.

  This test returns a reward if the latency is below a certain threshold for all
  steps in the episode.
  The latency is specified in milliseconds.
  
  This test ensures that the agent (including the entire runloop) can
  fulfill latency guarantees.
  """

  def __init__(self, latency_in_ms: str, episode_length: str, **kwargs):
    """Initializes a new LatencyTestCase.

    Args:
      latency_in_ms: latency threshold in milliseconds
      episode_length: the length of each episode
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    super().__init__(**kwargs)
    self._latency_in_ms = int(latency_in_ms)
    self._episode_length = int(episode_length)
    if self._latency_in_ms <= 0:
      raise ValueError("Latency threshold must be positive.")
    if self._episode_length <= 0:
      raise ValueError("Episode length must be positive.")
    self._timestamps = []

  def reset(self) -> dm_env.TimeStep:
    self._timestamps = [time.time()]
    return super().base_reset()

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    del action
    self._timestamps.append(time.time())
    # Reached episode end.
    if self.step_counter + 1 == self._episode_length:
      last_timestamp = self._timestamps[0]
      for timestamp in self._timestamps[1:]:
        if timestamp - last_timestamp > self._latency_in_ms / 1000.:
          return super().base_step(success=False, terminate=True)
        last_timestamp = timestamp
      return super().base_step(success=True, terminate=True)
    return super().base_step()


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  # Test cases correspond to 100Hz, 30Hz and 8Hz.
  # Each test runs for 128 timesteps by default, which corresponds to
  # multiple seconds for most framerates.
  return [f"latency@{n}@128" for n in [10, 34, 125]]
