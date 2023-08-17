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

"""Tests for tsuite."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import test_utils
from tsuite._src import tsuite


class TsuiteTest(test_utils.TSuiteTest):

  @parameterized.parameters([True, False])
  def test_list_test_tasks(self, include_broken):
    tsuite.list_test_tasks(
        self._action_spec,
        self._observation_spec,
        include_broken_env_tasks=include_broken)

  @parameterized.parameters([True, False])
  def test_tsuite_environment(self, verbose):
    test_tasks = tsuite.list_test_tasks(
        self._action_spec,
        self._observation_spec,
        include_broken_env_tasks=False)
    for test_task in test_tasks:
      with self.subTest(f'{test_task}'):
        env = tsuite.TSuiteEnvironment(
            test_task=test_task,
            action_spec=self._action_spec,
            observation_spec=self._observation_spec,
            verbose_logging=verbose)
        ts = env.reset()
        self.assertTrue(ts.first())
        action = ({'a': [np.array(1, dtype=np.int32)]},
                  np.array([-0.5, 0.5], dtype=np.float32),
                  np.ones((2, 3), dtype=np.float32))
        env.step(action)
        action = env.read_property(tsuite.PROPERTY_BEST_ACTION)
        env.step(action)
        action = env.read_property(tsuite.PROPERTY_RANDOM_ACTION)
        env.step(action)
        action = env.read_property(tsuite.PROPERTY_WORST_ACTION)
        env.step(action)


if __name__ == '__main__':
  absltest.main()
