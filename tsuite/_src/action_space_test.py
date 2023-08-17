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
"""Tests for action_space."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import action_space
from tsuite._src import agent
from tsuite._src import base
from tsuite._src import test_utils
from tsuite._src import tsuite

SUCCESS = base.SUCCESS
FAIL = base.FAIL


class ActionSpaceTest(test_utils.TSuiteTest):

  @parameterized.parameters(
      ('discrete@high', [SUCCESS, FAIL, FAIL]),
      ('discrete@low', [FAIL, FAIL, SUCCESS]),
      ('cont_up@high', [FAIL, FAIL, SUCCESS]),
      ('cont_up@low', [FAIL, SUCCESS, FAIL]),
      ('cont_left@high', [FAIL, SUCCESS, FAIL]),
      ('cont_left@low', [FAIL, FAIL, SUCCESS]),
      ('tensor_0@low', [FAIL, FAIL, SUCCESS]),
      ('tensor_0@high', [FAIL, SUCCESS, FAIL]),
      ('tensor_1@low', [FAIL, SUCCESS, FAIL]),
      ('tensor_1@high', [SUCCESS, FAIL, FAIL]),
      ('tensor_2@low', [SUCCESS, FAIL, FAIL]),
      ('tensor_2@high', [FAIL, FAIL, SUCCESS]),
      ('tensor_3@low', [SUCCESS, FAIL, FAIL]),
      ('tensor_3@high', [FAIL, SUCCESS, FAIL]),
      ('tensor_4@low', [FAIL, SUCCESS, FAIL]),
      ('tensor_4@high', [FAIL, FAIL, SUCCESS]),
      ('tensor_5@low', [FAIL, FAIL, SUCCESS]),
      ('tensor_5@high', [SUCCESS, FAIL, FAIL]),
  )
  def test_action_space(self, identifier, reward_sequence):
    env = tsuite.TSuiteEnvironment(
        f'action_space@{identifier}', self._action_spec, self._observation_spec)
    action = ({'a': [np.array(3, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32))
    self.assertIsNone(env.step(action).reward)
    self.assertEqual(env.step(action).reward, reward_sequence[0])
    action = ({'a': [np.array(2, dtype=np.int32)]},
              np.array([-0.9, 0.9], dtype=np.float32),
              np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32))
    self.assertIsNone(env.step(action).reward)
    self.assertEqual(env.step(action).reward, reward_sequence[1])
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([0.9, -0.9], dtype=np.float32),
              np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]], dtype=np.float32))
    self.assertIsNone(env.step(action).reward)
    self.assertEqual(env.step(action).reward, reward_sequence[2])

  @parameterized.parameters(
      ('discrete@high',),
      ('discrete@low',),
      ('cont_up@high',),
      ('cont_up@low',),
      ('cont_left@high',),
      ('cont_left@low',),
      ('tensor_0@low',),
      ('tensor_0@high',),
      ('tensor_1@low',),
      ('tensor_1@high',),
      ('tensor_2@low',),
      ('tensor_2@high',),
      ('tensor_3@low',),
      ('tensor_3@high',),
      ('tensor_4@low',),
      ('tensor_4@high',),
      ('tensor_5@low',),
      ('tensor_5@high',),
  )
  def test_action_space_with_best_action(self, identifier):
    env = tsuite.TSuiteEnvironment(
        f'action_space@{identifier}', self._action_spec, self._observation_spec)
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  @parameterized.parameters(
      ('discrete@high',),
      ('discrete@low',),
  )
  def test_with_agent(self, identifier):
    logs = agent.fit_agent_to_tsuite_task(
        f'action_space@{identifier}',
        early_stopping_mean_return=0.9)
    self.assertGreater(logs[-1]['value'], 0.9)

  def test_action_space_list_test_tasks(self):
    self.assertSetEqual(
        set(action_space.list_test_tasks(self._action_spec)),
        {'action_space@discrete@high', 'action_space@discrete@low',
         'action_space@cont_up@high', 'action_space@cont_up@low',
         'action_space@cont_left@high', 'action_space@cont_left@low',
         'action_space@tensor_0@low', 'action_space@tensor_0@high',
         'action_space@tensor_1@low', 'action_space@tensor_1@high',
         'action_space@tensor_2@low', 'action_space@tensor_2@high',
         'action_space@tensor_3@low', 'action_space@tensor_3@high',
         'action_space@tensor_4@low', 'action_space@tensor_4@high',
         'action_space@tensor_5@low', 'action_space@tensor_5@high'})


if __name__ == '__main__':
  absltest.main()
