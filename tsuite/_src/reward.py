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
"""Tests the reward processing."""

from collections.abc import Sequence

import dm_env

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if reward is correctly passed to the agent.

  This test always returns a reward regardless of the agent input.
  """

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    del action
    return super().base_step(
        success=True,
        terminate=True)


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  return ["reward"]
