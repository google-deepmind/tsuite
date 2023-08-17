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
"""Tests for causal."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import causal
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class CausalTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('text',),
      ('dvector',),
      ('float',),
  )
  def test_causalobservation_space_correct_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'causal@rgb@{identifier}@90',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays two episodes with the correct behaviour.
    for _ in range(2):
      timestep = env.reset()
      if np.mean(timestep.observation['rgb']) > 0.0:
        discrete_action = 3
      else:
        discrete_action = 0
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

  @parameterized.parameters(
      ('text',),
      ('dvector',),
      ('float',),
  )
  def test_causal_incorrect_behaviour(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'causal@rgb@{identifier}@90',
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

  def test_causal_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'causal@rgb@float@90',
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
        'causal@float@float_2@90',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_causal_list_test_tasks(self):
    self.assertSetEqual(
        set(causal.list_test_tasks(self._observation_spec)),
        {'causal@rgb@text@90', 'causal@rgb@text@99',
         'causal@rgb@text_length@90', 'causal@rgb@text_length@99',
         'causal@rgb@raw_text@90', 'causal@rgb@raw_text@99',
         'causal@rgb@float@90', 'causal@rgb@float@99',
         'causal@rgb@dvector@90', 'causal@rgb@dvector@99',
         'causal@text@rgb@90', 'causal@text@rgb@99',
         'causal@text@text_length@90', 'causal@text@text_length@99',
         'causal@text@raw_text@90', 'causal@text@raw_text@99',
         'causal@text@float@90', 'causal@text@float@99',
         'causal@text@dvector@90', 'causal@text@dvector@99',
         'causal@text_length@rgb@90', 'causal@text_length@rgb@99',
         'causal@text_length@text@90', 'causal@text_length@text@99',
         'causal@text_length@raw_text@90', 'causal@text_length@raw_text@99',
         'causal@text_length@float@90', 'causal@text_length@float@99',
         'causal@text_length@dvector@90', 'causal@text_length@dvector@99',
         'causal@raw_text@rgb@90', 'causal@raw_text@rgb@99',
         'causal@raw_text@text@90', 'causal@raw_text@text@99',
         'causal@raw_text@text_length@90', 'causal@raw_text@text_length@99',
         'causal@raw_text@float@90', 'causal@raw_text@float@99',
         'causal@raw_text@dvector@90', 'causal@raw_text@dvector@99',
         'causal@float@rgb@90', 'causal@float@rgb@99',
         'causal@float@text@90', 'causal@float@text@99',
         'causal@float@raw_text@90', 'causal@float@raw_text@99',
         'causal@float@text_length@90', 'causal@float@text_length@99',
         'causal@float@dvector@90', 'causal@float@dvector@99',
         'causal@dvector@rgb@90', 'causal@dvector@rgb@99',
         'causal@dvector@text@90', 'causal@dvector@text@99',
         'causal@dvector@raw_text@90', 'causal@dvector@raw_text@99',
         'causal@dvector@float@90', 'causal@dvector@float@99',
         'causal@dvector@text_length@90', 'causal@dvector@text_length@99',
         })


if __name__ == '__main__':
  absltest.main()
