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
"""Tests if the agent is robust against bad observations."""

from collections.abc import Sequence
import itertools

import dm_env
import numpy as np
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if the agent is robust against bad observations.

  This TestCase simulates a broken environment; it is reasonable for an agent
  to crash with a meaningful error message.

  The agent receives bad input for its observation. Depending on the mode, the
  agent receives:
    - `nan`: not a number,
    - `inf`: infinity,
    - `dtype`: the wrong dtype (int <-> float)
  """

  def __init__(self, observation_name: str, mode: str, **kwargs):
    kwargs["default_observation_name"] = observation_name
    super().__init__(**kwargs)

    if mode == "nan":
      def visitor(path, node):
        if path == self._default_observation_path:
          node = np.full_like(node, np.nan)
        return node
    elif mode == "inf":
      def visitor(path, node):
        if path == self._default_observation_path:
          node = np.full_like(node, np.inf)
        return node
    elif mode == "dtype":
      def visitor(path, node):
        if path == self._default_observation_path:
          if np.issubdtype(node.dtype, np.integer):
            node = node.astype(np.float32)
          else:
            node = node.astype(np.int32)
        return node
    else:
      raise ValueError(
          f"Unknown mode {mode} passed to bad_observation.TestCase.")

    self._observations = tree.map_structure_with_path(
        visitor, self.get_observation())

  def reset(self) -> dm_env.TimeStep:
    return super().base_reset(
        observation=self._observations)

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    del action
    return super().base_step(success=True,
                             terminate=True,
                             observation=self._observations)


def list_test_tasks(observation_spec: base.SpecsTree) -> Sequence[str]:
  """Returns available test-tasks of TestCase.

  Args:
    observation_spec: defines the observations consumed by the agent.
  """
  names = [spec.name for spec in tree.flatten(observation_spec)]
  modes = ["nan", "inf", "dtype"]
  return [f"bad_observation@{name}@{mode}"
          for name, mode in itertools.product(names, modes)]
