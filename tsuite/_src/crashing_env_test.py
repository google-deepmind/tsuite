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
"""Tests for crashing_env."""


from absl.testing import absltest
import numpy as np

from tsuite._src import crashing_env
from tsuite._src import test_utils
from tsuite._src import tsuite


class CrashingEnvTest(test_utils.TSuiteTest):

  def test_env_does_not_crash_with_probability_0(self):
    env = tsuite.TSuiteEnvironment(
        'crashing_env@0',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    timestep = env.reset()
    self.assertTrue(timestep.first())
    action = ({'a': [np.array(1, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    # Ensure that environment does not crash for 100 steps.
    for _ in range(100):
      env.step(action)

  def test_env_does_crash_with_probability_1(self):
    env = tsuite.TSuiteEnvironment(
        'crashing_env@100',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    with self.assertRaises(RuntimeError):
      env.reset()

  def test_crashing_env_list_test_tasks(self):
    self.assertSetEqual(
        set(crashing_env.list_test_tasks()),
        {'crashing_env@1'})


if __name__ == '__main__':
  absltest.main()
