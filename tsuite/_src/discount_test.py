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
"""Tests for discount."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import dm_env
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import discount
from tsuite._src import test_utils
from tsuite._src import tsuite

SUCCESS = base.SUCCESS


def _get_returns_for_n_step_episode(
    env: dm_env.Environment,
    n_steps: int,
    gamma: float) -> tuple[float, float]:
  env.reset()
  low_action = ({'a': [np.array(0, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
  high_action = ({'a': [np.array(3, dtype=np.int32)]},
                 np.array([-0.5, 0.5], dtype=np.float32),
                 np.ones((2, 3), dtype=np.float32))

  total_return = 0.0
  discounted_return = 0.0
  total_gamma = 1.0
  for i in range(n_steps):
    timestep = env.step(high_action if i+1 == n_steps else low_action)
    total_return += timestep.reward
    discounted_return += total_gamma * timestep.reward
    total_gamma *= gamma
  return total_return, discounted_return


class DiscountTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      (0.9,),
      (0.95,),
      (0.99,),
  )
  def test_discount(self, gamma):
    env = tsuite.TSuiteEnvironment(
        f'discount@{gamma}', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    r_nm1, dr_nm1 = _get_returns_for_n_step_episode(
        env, discount.N_OPTIMAL_STEPS-1, gamma)
    r_n, dr_n = _get_returns_for_n_step_episode(
        env, discount.N_OPTIMAL_STEPS, gamma)
    r_np1, dr_np1 = _get_returns_for_n_step_episode(
        env, discount.N_OPTIMAL_STEPS+1, gamma)
    r_np2, dr_np2 = _get_returns_for_n_step_episode(
        env, discount.N_OPTIMAL_STEPS+2, gamma)
    # Reward is close to one for the optimal steps.
    self.assertAlmostEqual(r_n, 1.0, places=3)
    self.assertLess(r_nm1, 1.0)
    self.assertGreater(r_np1, 1.0)
    self.assertGreater(r_np2, r_np1)
    # Discounted reward has a maximum at the optimal steps.
    self.assertLess(dr_nm1, dr_n)
    self.assertLessEqual(dr_np1, dr_n)
    self.assertLess(dr_np2, dr_n)

  def test_discount_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'discount@0.99', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      ts = env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertIsNone(ts.reward)
      total_return = 0.0
      while not ts.last():
        action = env.read_property(tsuite.PROPERTY_BEST_ACTION)
        ts = env.step(action)
        logging.info(ts.reward)
        total_return += ts.reward
      self.assertAlmostEqual(total_return, SUCCESS, places=2)

  def test_with_agent(self):
    logs = agent.fit_agent_to_tsuite_task(
        'discount@0.99',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_zero_discount_list_test_tasks(self):
    self.assertSetEqual(
        set(discount.list_test_tasks()),
        {'discount@0.99'})

if __name__ == '__main__':
  absltest.main()
