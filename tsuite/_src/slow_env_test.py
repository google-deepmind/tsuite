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
"""Tests for slow_env."""

import time

from absl.testing import absltest
import numpy as np

from tsuite._src import base
from tsuite._src import slow_env
from tsuite._src import test_utils
from tsuite._src import tsuite

SUCCESS = base.SUCCESS
FAIL = base.FAIL


class SlowEnvTest(test_utils.TSuiteTest):

  def test_slow_env_correct_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'slow_env@100',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    start_time = time.time()
    # Plays one episode with the correct behaviour.
    timestep = env.reset()
    # The following action sequence corresponds to the "secret" in the
    # overfit.TestCase class.
    for discrete_action in [0, 3, 3, 0]:
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
    # With the slow_env, each reset and step takes at least 100 ms, hence the
    # test should take at least 500 ms.
    self.assertGreater(time.time() - start_time, 0.5)
    self.assertEqual(timestep.reward, SUCCESS)

  def test_slow_env_incorrect_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'slow_env@100',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    start_time = time.time()
    # Plays one episode with the incorrect behaviour.
    timestep = env.reset()
    # The following action sequence does not corresponds to the "secret" in the
    # overfit.TestCase class.
    for discrete_action in [0, 3, 3, 2]:
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
    # With the slow_env, each reset and step takes at least 100 ms, hence the
    # test should take at least 500 ms.
    self.assertGreater(time.time() - start_time, 0.5)
    self.assertEqual(timestep.reward, FAIL)

  def test_slow_env_list_test_tasks(self):
    self.assertSetEqual(
        set(slow_env.list_test_tasks()),
        {'slow_env@500'})


if __name__ == '__main__':
  absltest.main()
