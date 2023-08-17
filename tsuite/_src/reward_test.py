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
"""Tests for reward."""

from absl.testing import absltest
import numpy as np

from tsuite._src import base
from tsuite._src import reward
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS


class RewardTest(test_utils.TSuiteTest):

  def test_reward(self):
    env = tsuite.TSuiteEnvironment(
        'reward', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    timestep = env.reset()
    self.assertTrue(timestep.first())
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    timestep = env.step(action)
    self.assertEqual(timestep.reward, SUCCESS)

  def test_reward_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'reward', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  def test_reward_list_test_tasks(self):
    self.assertSetEqual(
        set(reward.list_test_tasks()),
        {'reward'})


if __name__ == '__main__':
  absltest.main()
