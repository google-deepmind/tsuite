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
"""Tests for bad_timestep."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import bad_timestep
from tsuite._src import test_utils
from tsuite._src import tsuite


class BadTimestepTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('discount', 'negative', -1.0),
      ('discount', 'oor', 1.1),
      ('discount', 'nan', np.nan),
      ('discount', 'inf', np.inf),
      ('reward', 'negative', -1.0),
      ('reward', 'nan', np.nan),
      ('reward', 'inf', np.inf),
      ('step_type', 'negative', -1),
      ('step_type', 'oor', 3),
      ('step_type', 'nan', np.nan),
      ('step_type', 'inf', np.inf),
  )
  def test_bad_timestep(self, identifier, mode, expected):
    env = tsuite.TSuiteEnvironment(
        f'bad_timestep@{identifier}@{mode}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    env.reset()
    action = ({'a': [np.array(2, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    timestep = env.step(action)
    var = getattr(timestep, identifier)
    self.assertTrue(expected is var or expected == var)

  def test_bad_timestep_list_test_tasks(self):
    self.assertSetEqual(
        set(bad_timestep.list_test_tasks()),
        {'bad_timestep@discount@nan', 'bad_timestep@discount@inf',
         'bad_timestep@discount@oor', 'bad_timestep@discount@negative',
         'bad_timestep@reward@nan', 'bad_timestep@reward@inf',
         'bad_timestep@reward@negative',
         'bad_timestep@step_type@nan', 'bad_timestep@step_type@inf',
         'bad_timestep@step_type@oor', 'bad_timestep@step_type@negative',})


if __name__ == '__main__':
  absltest.main()
