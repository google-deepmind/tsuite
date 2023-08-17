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
"""Tests for thread_safety."""

import concurrent.futures

from absl.testing import absltest
import numpy as np

from tsuite._src import base
from tsuite._src import test_utils
from tsuite._src import thread_safety
from tsuite._src import tsuite


SUCCESS = base.SUCCESS


class ThreadSafetyTest(test_utils.TSuiteTest):

  def _run_single_env(self, n_steps: int):
    env = tsuite.TSuiteEnvironment(
        'thread_safety', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')

    timestep = env.reset()
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    for _ in range(n_steps):
      timestep = env.step(action)
    return timestep

  def test_thread_safety_single_env(self):
    timestep = self._run_single_env(n_steps=1)
    self.assertEqual(timestep.reward, SUCCESS)

  def test_thread_safety_multi_env(self):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
      futures = [executor.submit(self._run_single_env, n_steps=20)
                 for _ in range(20)]
      with self.assertRaises(RuntimeError):
        for f in futures:
          f.result()

  def test_thread_safety_list_test_tasks(self):
    self.assertSetEqual(
        set(thread_safety.list_test_tasks()),
        {'thread_safety'})

if __name__ == '__main__':
  absltest.main()
