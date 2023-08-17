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
"""Tests for zero_discount."""

from absl.testing import absltest
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import test_utils
from tsuite._src import tsuite
from tsuite._src import zero_discount


SUCCESS = base.SUCCESS


class ZeroDiscountTest(test_utils.TSuiteTest):

  def test_zero_discount(self):
    env = tsuite.TSuiteEnvironment(
        'zero_discount', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    env.reset()
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    timestep = env.step(action)
    self.assertEqual(timestep.discount, 0.0)
    timestep = env.step(action)
    self.assertEqual(timestep.reward, SUCCESS)
    self.assertEqual(timestep.discount, 0.0)

  def test_discount_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'zero_discount', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  def test_with_agent(self):
    logs = agent.fit_agent_to_tsuite_task(
        'zero_discount',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_zero_discount_list_test_tasks(self):
    self.assertSetEqual(
        set(zero_discount.list_test_tasks()),
        {'zero_discount'})

if __name__ == '__main__':
  absltest.main()
