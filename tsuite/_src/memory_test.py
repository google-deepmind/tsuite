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
"""Tests for memory."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import memory
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class MemoryTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('1',),
      ('2',),
      ('10',),
  )
  def test_memory_correct_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'memory@{identifier}', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      if timestep.observation['float'] > 0.0:
        discrete_action = 3
      else:
        discrete_action = 0
      while not timestep.last():
        action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                  np.array([-0.5, 0.5], dtype=np.float32),
                  np.ones((2, 3), dtype=np.float32))
        timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(
      ('1',),
      ('2',),
      ('10',),
  )
  def test_memory_incorrect_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'memory@{identifier}', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')

    # Plays two episodes with the incorrect behaviour.
    for _ in range(2):
      timestep = env.reset()
      while not timestep.last():
        action = ({'a': [np.array(2, dtype=np.int32)]},
                  np.array([-0.5, 0.5], dtype=np.float32),
                  np.ones((2, 3), dtype=np.float32))
        timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

  def test_memory_incorrect_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'memory@5',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      timestep = env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertIsNone(timestep.reward)
      while not timestep.last():
        timestep = env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(
      ('memory@1',),
      ('memory@2',),
  )
  def test_with_agent(self, identifier):
    logs = agent.fit_agent_to_tsuite_task(
        f'{identifier}',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_memory_list_test_tasks(self):
    self.assertSetEqual(
        set(memory.list_test_tasks()),
        {'memory@0', 'memory@1', 'memory@2', 'memory@3', 'memory@4',
         'memory@5', 'memory@6', 'memory@7', 'memory@8', 'memory@9'})


if __name__ == '__main__':
  absltest.main()
