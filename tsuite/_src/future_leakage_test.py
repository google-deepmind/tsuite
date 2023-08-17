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
"""Tests for future_leakage."""

from absl.testing import absltest
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import future_leakage
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class FutureLeakageTest(test_utils.TSuiteTest):

  def test_future_leakage_correct_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'future_leakage',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episode with the correct behaviour.
    for discrete_action in [0, 3]:
      env.reset()
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  def test_future_leakage_incorrect_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'future_leakage',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays four episode with the incorrect behaviour.
    for discrete_action in [3, 0, 1, 2]:
      env.reset()
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

  def test_future_leakage_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'future_leakage',
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
        'future_leakage',
        n_updates=50)
    self.assertLess(logs[-1]['value'], 0.6)
    self.assertGreater(logs[-1]['value'], 0.4)

  def test_future_leakage_list_test_tasks(self):
    self.assertSetEqual(
        set(future_leakage.list_test_tasks()),
        {'future_leakage'})


if __name__ == '__main__':
  absltest.main()
