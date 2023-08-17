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
"""Tests for language."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import base
from tsuite._src import language
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class LanguageTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('raw_text', 'content',),
      ('raw_text', 'length',),
  )
  def test_language_correct_behaviour(self, observation_name, mode):
    env = tsuite.TSuiteEnvironment(
        f'language@{observation_name}@{mode}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      discrete_action = 2
      if mode == 'content':
        if timestep.observation[observation_name] == 'left':
          discrete_action = 3
        if timestep.observation[observation_name] == 'right':
          discrete_action = 0
      if mode == 'length':
        if timestep.observation[observation_name] == 'no no':
          discrete_action = 3
        if timestep.observation[observation_name] == 'no no no':
          discrete_action = 0
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(('raw_text', 'content',), ('raw_text', 'length',))
  def test_language_incorrect_behaviour(self, observation_name, mode):
    env = tsuite.TSuiteEnvironment(
        f'language@{observation_name}@{mode}',
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

  def test_language_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'language@raw_text@length',
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

  def test_language_list_test_tasks(self):
    self.assertSetEqual(
        set(language.list_test_tasks(self._observation_spec)),
        {'language@raw_text@content', 'language@raw_text@length'})


if __name__ == '__main__':
  absltest.main()
