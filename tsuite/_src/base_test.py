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
"""Tests for base."""

import enum

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np

from tsuite._src import base


class _StringLikeEnum(str, enum.Enum):
  OBS = 'obs'


class _StringLikeEnum2(str, enum.Enum):
  __str__ = str.__str__
  OBS = 'obs'


class BaseTest(parameterized.TestCase):

  def test_set_names_in_spec(self):
    spec = {
        'a': [specs.DiscreteArray(num_values=4)],
        'b': specs.DiscreteArray(num_values=4, name='b')
    }
    expected_spec = {
        'a': [specs.DiscreteArray(num_values=4, name="('a', 0)")],
        'b': specs.DiscreteArray(num_values=4, name='b')
    }
    self.assertDictEqual(base.set_names_in_spec(spec), expected_spec)

  def test_get_action(self):
    action_spec = {
        'a': specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1,
            name='a'),
        'b': specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1,
            name='b1|b2'),
        'c': specs.BoundedArray(
            shape=(2, 1), dtype=np.float32, minimum=-1, maximum=1,
            name='c'),
        'd': specs.BoundedArray(
            shape=(2,), dtype=np.int32, minimum=-1, maximum=1,
            name='d'),
        'e': specs.StringArray((), name='e'),
        'f': specs.Array(shape=(2,), dtype=bool, name='f'),
    }
    action = {
        'a': np.array([-1.0, 1.0]),
        'b': np.array([0.0, 1.0]),
        'c': np.array([[-1.0], [1.0]]),
        'd': np.array([-1, 1]),
        'e': np.array(base.ExternalStringAction.HIGH.value, dtype=np.str_),
        'f': np.array([True, False]),
    }
    observation_spec = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='obs')
    t1 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='a_0')
    self.assertEqual(t1.map_external_to_internal_action(action),
                     base.InternalAction.LOW)
    t2 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='a_1')
    self.assertEqual(t2.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t3 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='b1')
    self.assertEqual(t3.map_external_to_internal_action(action),
                     base.InternalAction.NOOP)
    t4 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='b2')
    self.assertEqual(t4.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t5 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='c_0')
    self.assertEqual(t5.map_external_to_internal_action(action),
                     base.InternalAction.LOW)
    t6 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='c_1')
    self.assertEqual(t6.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t7 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='d_0')
    self.assertEqual(t7.map_external_to_internal_action(action),
                     base.InternalAction.LOW)
    t8 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='d_1')
    self.assertEqual(t8.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t9 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='e')
    self.assertEqual(t9.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t10 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='f_0')
    self.assertEqual(t10.map_external_to_internal_action(action),
                     base.InternalAction.HIGH)
    t11 = base.TestCase(
        action_spec=action_spec,
        observation_spec=observation_spec,
        default_action_name='f_1')
    self.assertEqual(t11.map_external_to_internal_action(action),
                     base.InternalAction.LOW)

  def test_get_inverse_action(self):
    action_spec = {
        'a': specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='a'),
        'b': specs.DiscreteArray(num_values=4, name='b'),
        'c': specs.BoundedArray(
            shape=(2, 3), dtype=np.float32, minimum=-1, maximum=1,
            name='c'),
        'd': specs.StringArray((), name='d'),
        'e': specs.Array(shape=(2,), dtype=bool, name='e'),
    }
    observation_spec = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='obs')
    for default_action_name in ['a_0', 'a_1', 'b', 'c_0', 'c_1', 'c_2',
                                'c_3', 'c_4', 'c_5', 'd']:
      test_case = base.TestCase(
          action_spec=action_spec,
          observation_spec=observation_spec,
          default_action_name=default_action_name)
      for action in [base.InternalAction.LOW, base.InternalAction.HIGH,
                     base.InternalAction.NOOP]:
        with self.subTest(f'{default_action_name}_{action}'):
          self.assertEqual(
              test_case.map_external_to_internal_action(
                  test_case.map_internal_to_external_action(action)),
              action)

    # Test boolean case.
    for default_action_name in ['e_0', 'e_1']:
      test_case = base.TestCase(
          action_spec=action_spec,
          observation_spec=observation_spec,
          default_action_name=default_action_name)
      for action in [base.InternalAction.LOW, base.InternalAction.HIGH]:
        with self.subTest(f'{default_action_name}_{action}'):
          self.assertEqual(
              test_case.map_external_to_internal_action(
                  test_case.map_internal_to_external_action(action)),
              action)

  @parameterized.parameters(
      dict(spec=specs.BoundedArray(
          shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='obs'),
           signal=np.array([1, 1]), no_signal=np.array([-1, -1])),
      dict(spec=specs.Array(shape=(2,), dtype=np.bytes_, name='obs'),
           signal=np.array([b'1', b'1']), no_signal=np.array([b'0', b'0'])),
      dict(spec=specs.Array(shape=(2,), dtype=np.str_, name='obs'),
           signal=np.array(['1', '1']), no_signal=np.array(['0', '0'])),
      dict(spec=specs.Array(shape=(), dtype=np.float32, name='obs'),
           signal=np.array(1), no_signal=np.array(0)),
      dict(spec=specs.Array(shape=(2,), dtype=np.float32, name=b'obs'),
           signal=np.array([1, 1]), no_signal=np.array([0, 0])),
      dict(spec=specs.Array(
          shape=(2,), dtype=np.float32, name='obs'),
           signal=np.array([1, 1]), no_signal=np.array([0, 0])),
      dict(spec=specs.Array(
          shape=(2,), dtype=np.float32, name=_StringLikeEnum.OBS),
           signal=np.array([1, 1]), no_signal=np.array([0, 0])),
      dict(spec=specs.Array(
          shape=(2,), dtype=np.float32, name=_StringLikeEnum2.OBS),
           signal=np.array([1, 1]), no_signal=np.array([0, 0])),
  )
  def test_get_signal_injector_visitor_fn(
      self, spec: specs.Array, signal: np.ndarray, no_signal: np.ndarray):
    # Spec names are always converted to string in tsuite.
    name = str(spec.name)
    with self.subTest('InjectSignal'):
      fn = base.make_signal_injector_visitor_fn([name])
      np.testing.assert_array_equal(fn(spec), signal)
    with self.subTest('NotInjectSignal'):
      fn = base.make_signal_injector_visitor_fn([])
      np.testing.assert_array_equal(fn(spec), no_signal)


if __name__ == '__main__':
  absltest.main()
