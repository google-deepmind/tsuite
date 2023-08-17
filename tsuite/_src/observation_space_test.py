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
"""Tests for observation_space."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import observation_space
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class ObservationSpaceTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('rgb',),
      ('text',),
      ('dvector',),
      ('float',),
  )
  def test_observation_space_correct_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'observation_space@{identifier}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      if identifier == 'text':
        if np.mean(timestep.observation[identifier][0]) > 0.0:
          discrete_action = 3
        else:
          discrete_action = 0
      else:
        if np.mean(timestep.observation[identifier]) > 0.0:
          discrete_action = 3
        else:
          discrete_action = 0
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(
      ('rgb',),
      ('text',),
      ('dvector',),
      ('float',),
  )
  def test_observation_space_incorrect_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'observation_space@{identifier}',
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

  def test_observation_space_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'observation_space@rgb',
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
        'observation_space@float',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_observation_space_list_test_tasks(self):
    self.assertSetEqual(
        set(observation_space.list_test_tasks(self._observation_spec)),
        {'observation_space@rgb', 'observation_space@text',
         'observation_space@text_length', 'observation_space@raw_text',
         'observation_space@dvector', 'observation_space@float'})


if __name__ == '__main__':
  absltest.main()
