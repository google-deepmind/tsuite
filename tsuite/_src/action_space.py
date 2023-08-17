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
"""Tests the action space of the agent."""

from collections.abc import Sequence
import itertools

import dm_env
import numpy as np
import tree

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests the action space of the agent.

  The agent receives a reward if its able to output a high / low value
  for a specific action.

  This test ensures that the agent can take advantage of the entire action
  space.
  """

  def __init__(self, action_name: str, mode: str, **kwargs):
    """Initializes a new ActionSpaceTestCase.

    Args:
      action_name: name of the tested action.
      mode: defines which output receives the reward, either high or low.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    kwargs["default_action_name"] = action_name
    super().__init__(**kwargs)
    if mode == "high":
      self._expected = base.InternalAction.HIGH
    elif mode == "low":
      self._expected = base.InternalAction.LOW
    else:
      raise ValueError(f"Unknown mode {mode} passed to ActionSpaceTestCase.")

  def step(self, action) -> dm_env.TimeStep:
    internal_action = self.map_external_to_internal_action(action)
    return super().base_step(success=internal_action == self._expected,
                             terminate=True)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    return self._expected


def list_test_tasks(action_spec: base.ActionSpecsTree) -> Sequence[str]:
  """Returns available test-tasks of TestCase.

  Args:
    action_spec: defines the action space of the agent.
  """
  names = []
  for node in tree.flatten(action_spec):
    n_dimensions = len(node.shape)
    if n_dimensions == 0:
      names.append(node.name)
    elif "|" in node.name:
      names += list(node.name.split("|"))
    else:
      names += [node.name + f"_{index}"
                for index in range(np.prod(node.shape))]
  modes = ["high", "low"]
  return [f"action_space@{name}@{mode}"
          for name, mode in itertools.product(names, modes)]
