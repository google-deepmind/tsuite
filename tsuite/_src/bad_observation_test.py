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
"""Tests for bad_observation."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import bad_observation
from tsuite._src import test_utils
from tsuite._src import tsuite


class BadObservationTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('rgb', 'nan'),
      ('rgb', 'inf'),
      ('rgb', 'dtype'),
      ('dvector', 'dtype'),
      ('float', 'nan'),
      ('float', 'inf'),
      ('float', 'dtype'),
  )
  def test_bad_observations(self, identifier, mode):
    env = tsuite.TSuiteEnvironment(
        f'bad_observation@{identifier}@{mode}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    timestep = env.reset()
    if mode == 'nan':
      self.assertTrue(np.all(np.isnan(timestep.observation[identifier])))
    elif mode == 'inf':
      self.assertTrue(np.all(np.isinf(timestep.observation[identifier])))
    elif mode == 'dtype':
      if identifier in ['rgb', 'float']:
        self.assertEqual(timestep.observation[identifier].dtype, np.int32)
      else:
        self.assertEqual(timestep.observation[identifier].dtype, np.float32)

    action = ({'a': [np.array(2, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    timestep = env.step(action)
    if mode == 'nan':
      self.assertTrue(np.all(np.isnan(timestep.observation[identifier])))
    elif mode == 'inf':
      self.assertTrue(np.all(np.isinf(timestep.observation[identifier])))
    elif mode == 'dtype':
      if identifier in ['rgb', 'float']:
        self.assertEqual(timestep.observation[identifier].dtype, np.int32)
      else:
        self.assertEqual(timestep.observation[identifier].dtype, np.float32)

  def test_bad_observations_list_test_tasks(self):
    self.assertSetEqual(
        set(bad_observation.list_test_tasks(self._observation_spec)),
        {'bad_observation@rgb@nan', 'bad_observation@text@nan',
         'bad_observation@text_length@nan',
         'bad_observation@dvector@nan', 'bad_observation@float@nan',
         'bad_observation@rgb@inf', 'bad_observation@text@inf',
         'bad_observation@text_length@inf',
         'bad_observation@dvector@inf', 'bad_observation@float@inf',
         'bad_observation@rgb@dtype', 'bad_observation@text@dtype',
         'bad_observation@text_length@dtype',
         'bad_observation@dvector@dtype', 'bad_observation@float@dtype',
         'bad_observation@raw_text@inf', 'bad_observation@raw_text@nan',
         'bad_observation@raw_text@dtype'})


if __name__ == '__main__':
  absltest.main()
