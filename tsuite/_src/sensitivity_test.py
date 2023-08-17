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
"""Tests for sensitivity."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import sensitivity
from tsuite._src import test_utils
from tsuite._src import tsuite

SUCCESS = base.SUCCESS
FAIL = base.FAIL


class SensitivitySpaceTest(test_utils.TSuiteTest):

  # Test only float observations here, because integer observations cannot
  # represent numbers smaller than 1, hence n<0 will fail.
  @parameterized.product(
      identifier=('rgb', 'float'),
      n=(-2, -1, 0, 1, 2),
  )
  def test_sensitivity_correct_behaviour(self, identifier, n):
    env = tsuite.TSuiteEnvironment(
        f'sensitivity@{identifier}@{n}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      if np.mean(timestep.observation[identifier]) > 0.0:
        discrete_action = 3
      else:
        discrete_action = 0
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.product(
      identifier=('rgb', 'dvector', 'text', 'float'),
      n=(-2, 1, 0, 1, 2),
  )
  def test_sensitivity_incorrect_behaviour(self, identifier, n):
    env = tsuite.TSuiteEnvironment(
        f'sensitivity@{identifier}@{n}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the incorrect behaviour.
    for _ in range(2):
      timestep = env.reset()
      self.assertTrue(timestep.first())
      action = ({'a': [np.array(2, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

  @parameterized.product(
      identifier=('rgb', 'dvector', 'text', 'float'),
      n=(-2, 1, 0, 1, 2),
  )
  def test_sensitivity_with_best_action(self, identifier, n):
    env = tsuite.TSuiteEnvironment(
        f'sensitivity@{identifier}@{n}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  def test_with_agent(self):
    logs = agent.fit_agent_to_tsuite_task(
        'sensitivity@float@0',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_sensitivity_list_test_tasks(self):
    self.assertSetEqual(
        set(sensitivity.list_test_tasks(self._observation_spec)),
        {'sensitivity@rgb@-1', 'sensitivity@rgb@0', 'sensitivity@rgb@1',
         'sensitivity@text_length@-1', 'sensitivity@text_length@0',
         'sensitivity@text_length@1',
         'sensitivity@text@-1', 'sensitivity@text@0',
         'sensitivity@text@1',
         'sensitivity@dvector@-1', 'sensitivity@dvector@0',
         'sensitivity@dvector@1',
         'sensitivity@float@-1', 'sensitivity@float@0', 'sensitivity@float@1'})


if __name__ == '__main__':
  absltest.main()
