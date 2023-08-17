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
"""Tests the visual capabilities of the agent."""

from collections.abc import Sequence
import itertools

import dm_env
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the visual capabilities of the agent.

  The agent receives a reward if it's able to output a high / low value for the
  default action depending on the value of an image-like observation. Depending
  on the mode, the agent has to be able to distinguish:
    - `color`: a signal in the red from a signal in the green channel.
    - `size`: a signal in the small square from a signal in a large square.
    - `vertical_position`: a signal in the upper half from a signal in the lower
      half of the image.
    - `horizontal_position`: a signal in the left half from a signal in the
      right half of the image.
  """

  def __init__(self, observation_name: str, mode: str, **kwargs):
    kwargs["default_observation_name"] = observation_name
    super().__init__(**kwargs)

    if mode == "color":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          # Sets red channel to 1.
          node[:, :, 0] = 1
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          # Sets green channel to 1.
          node[:, :, 1] = 1
        return node
    elif mode == "size":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          # Creates a large square with 1.
          width_div_2 = int(node.shape[0] / 2)
          height_div_2 = int(node.shape[1] / 2)
          width_div_4 = int(width_div_2 / 2)
          height_div_4 = int(height_div_2 / 2)
          node[width_div_2-width_div_4:width_div_2+width_div_4,
               height_div_2-height_div_4:height_div_2+height_div_4, 0] = 1
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          # Creates a small square with 1.
          width_div_2 = int(node.shape[0] / 2)
          height_div_2 = int(node.shape[1] / 2)
          width_div_8 = int(width_div_2 / 4)
          height_div_8 = int(height_div_2 / 4)
          node[width_div_2-width_div_8:width_div_2+width_div_8,
               height_div_2-height_div_8:height_div_2+height_div_8, 0] = 1
        return node
    elif mode == "vertical_position":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          # Sets upper half to 1.
          width_div_2 = int(node.shape[0] / 2)
          node[:width_div_2, :, 0] = 1
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          # Sets lower half to 1.
          width_div_2 = int(node.shape[0] / 2)
          node[width_div_2:, :, 0] = 1
        return node
    elif mode == "horizontal_position":
      def high_visitor(path, node):
        if path == self._default_observation_path:
          # Sets upper half to 1.
          height_div_2 = int(node.shape[1] / 2)
          node[:, :height_div_2, 0] = 1
        return node
      def low_visitor(path, node):
        if path == self._default_observation_path:
          # Sets lower half to 1.
          height_div_2 = int(node.shape[1] / 2)
          node[:, height_div_2:, 0] = 1
        return node
    else:
      raise ValueError(f"Unknown mode {mode} passed to VisualTestCase.")

    self._visual_observations = [
        tree.map_structure_with_path(low_visitor, self.get_observation()),
        tree.map_structure_with_path(high_visitor, self.get_observation())
    ]

  def reset(self) -> dm_env.TimeStep:
    return super().base_reset(
        observation=self._visual_observations[self.episode_counter % 2])

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
    # Shape indicates RGB or RGBA observation.
    if len(node.shape) == 3 and node.shape[2] in [3, 4]:
      names.append(node.name)
  modes = ["color", "size", "vertical_position", "horizontal_position"]
  return [f"visual@{name}@{mode}"
          for name, mode in itertools.product(names, modes)]


