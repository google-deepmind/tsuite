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
"""Tests the language capabilities of the agent."""

from collections.abc import Sequence
import itertools

import dm_env
import numpy as np
import tree


from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the language capabilities of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of a text-like observation. Depending
  on the mode, the agent has to be able to distinguish:
    - `content`: the word "left" from the word "right",
    - `length`: the text "no no" from the text "no no no".
  """

  def __init__(self, observation_name: str, mode: str, **kwargs):
    kwargs["default_observation_name"] = observation_name
    super().__init__(**kwargs)

    if mode == "content":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          node = np.broadcast_to(np.array("left", np.str_), node.shape)
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          node = np.broadcast_to(np.array("right", np.str_), node.shape)
        return node
    elif mode == "length":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          node = np.broadcast_to(np.array("no no", np.str_), node.shape)
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          node = np.broadcast_to(np.array("no no no", np.str_), node.shape)
        return node
    else:
      raise ValueError(f"Unknown mode {mode} passed to LanguageTestCase.")

    self._text_observations = [
        tree.map_structure_with_path(low_visitor, self.get_observation()),
        tree.map_structure_with_path(high_visitor, self.get_observation())
    ]

  def reset(self) -> dm_env.TimeStep:
    return super().base_reset(
        observation=self._text_observations[self.episode_counter % 2])

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    internal_action = self.map_external_to_internal_action(action)
    return super().base_step(
        success=internal_action == self.best_next_internal_action(),
        terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    signal = (self.episode_counter % 2) == 0
    return base.InternalAction.HIGH if signal else base.InternalAction.LOW


def list_test_tasks(observation_spec: base.SpecsTree) -> Sequence[str]:
  """Returns available test-tasks of TestCase.

  Args:
    observation_spec: defines the observations consumed by the agent.
  """
  names = []
  for node in tree.flatten(observation_spec):
    # Dtype indicates text observation.
    if node.dtype.type in [np.bytes_, np.str_]:
      names.append(node.name)
  modes = ["content", "length"]
  return [f"language@{name}@{mode}"
          for name, mode in itertools.product(names, modes)]


