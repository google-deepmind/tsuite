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
"""Tests if the agent is robust against bad timesteps."""

from collections.abc import Sequence
import itertools

import dm_env
import numpy as np

from tsuite._src import base


class TestCase(base.TestCase):
  """Tests if the agent is robust against bad timesteps.

  This TestCase simulates a broken environment; it is reasonable for an agent
  to crash with a meaningful error message.

  The agent receives bad input for the timestep variables `discount`, `reward`,
  `step_type`. Depending on the mode, the agent receives:
    - `negative`: a negative value,
    - `oor`: a finite non-negative out of range value (only available for
       discount and step_type),
    - `nan`: not a number,
    - `inf`: an infinite value.
  """

  def __init__(self, timestep_var: str, mode: str, **kwargs):
    super().__init__(**kwargs)

    if mode == "nan":
      self._timestep_update = {timestep_var: np.nan}
    elif mode == "inf":
      self._timestep_update = {timestep_var: np.inf}
    elif mode == "negative":
      # Integer type timestep variables
      if timestep_var in ["step_type"]:
        self._timestep_update = {timestep_var: -1}
      else:
        self._timestep_update = {timestep_var: -1.0}
    elif mode == "oor":
      if timestep_var == "discount":
        self._timestep_update = dict(discount=1.1)
      elif timestep_var == "step_type":
        self._timestep_update = dict(step_type=3)
      # In all other cases there is no finite non-negative out-of-range value.
      else:
        raise ValueError(
            f"Timestep variable {timestep_var} does not support oor mode.")
    else:
      raise ValueError(f"Unknown mode {mode} passed to bad_timestep.TestCase.")

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    del action
    if self.step_counter >= 1:
      timestep = super().base_step(success=True, terminate=True)
    else:
      timestep = super().base_step(success=False, terminate=False)
      timestep = timestep._replace(**self._timestep_update)
    return timestep


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase."""
  names = ["discount", "reward", "step_type"]
  modes = ["nan", "inf", "negative"]
  tasks = [f"bad_timestep@{name}@{mode}"
           for name, mode in itertools.product(names, modes)]
  tasks += ["bad_timestep@discount@oor", "bad_timestep@step_type@oor"]
  return tasks
