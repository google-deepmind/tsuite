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
"""Tests for visual."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import base
from tsuite._src import test_utils
from tsuite._src import tsuite
from tsuite._src import visual


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class VisualTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('rgb', 'color',),
      ('rgb', 'vertical_position',),
      ('rgb', 'horizontal_position',),
      ('rgb', 'size',),
  )
  def test_visual_correct_behaviour(self, observation_name, mode):
    env = tsuite.TSuiteEnvironment(
        f'visual@{observation_name}@{mode}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      discrete_action = 2
      if mode == 'color':
        if np.mean(timestep.observation[observation_name][:, :, 0]) > 0.0:
          discrete_action = 3
        if np.mean(timestep.observation[observation_name][:, :, 1]) > 0.0:
          discrete_action = 0
      if mode == 'size':
        if np.mean(timestep.observation[observation_name][:3, :, 0]) > -1.0:
          discrete_action = 3
        else:
          discrete_action = 0
      if mode == 'vertical_position':
        if np.mean(timestep.observation[observation_name][:4, :, 0]) > 0.0:
          discrete_action = 3
        if np.mean(timestep.observation[observation_name][4:, :, 0]) > 0.0:
          discrete_action = 0
      if mode == 'horizontal_position':
        if np.mean(timestep.observation[observation_name][:, :4, 0]) > 0.0:
          discrete_action = 3
        if np.mean(timestep.observation[observation_name][:, 4:, 0]) > 0.0:
          discrete_action = 0
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(
      ('rgb', 'color',),
      ('rgb', 'vertical_position',),
      ('rgb', 'horizontal_position',),
      ('rgb', 'size',),
  )
  def test_visual_incorrect_behaviour(self, observation_name, mode):
    env = tsuite.TSuiteEnvironment(
        f'visual@{observation_name}@{mode}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the incorrect behaviour.
    for _ in range(2):
      env.reset()
      action = ({'a': [np.array(2, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

  def test_visual_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'visual@rgb@size',
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

  def test_visual_list_test_tasks(self):
    self.assertSetEqual(
        set(visual.list_test_tasks(self._observation_spec)),
        {'visual@rgb@color', 'visual@rgb@size', 'visual@rgb@vertical_position',
         'visual@rgb@horizontal_position'})


if __name__ == '__main__':
  absltest.main()
