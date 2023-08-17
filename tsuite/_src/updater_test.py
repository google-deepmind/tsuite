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
"""Tests for updater."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tsuite._src import updater


class UpdaterTest(parameterized.TestCase):

  def test_updater(self):
    y_train = np.random.randint(2, size=1000)
    x_train = np.random.normal(y_train, 0.5, size=1000)[..., None]
    y_test = np.random.randint(2, size=1000)
    x_test = np.random.normal(y_test, 0.5, size=1000)[..., None]

    def network(x: chex.Array) -> chex.Array:
      return hk.nets.MLP(output_sizes=[16, 2])(x)

    def loss(x: chex.Array, y: chex.Array) -> chex.Array:
      return jnp.mean(optax.softmax_cross_entropy(
          logits=network(x), labels=jax.nn.one_hot(y, 2)))

    def accuracy(x: chex.Array, y: chex.Array) -> chex.Array:
      return jnp.mean(jnp.argmax(network(x), axis=1) == y)

    my_updater = updater.Updater(
        optimizer=optax.adam(learning_rate=1e-3),
        loss=loss,
        rng_key=jax.random.PRNGKey(42),
        metrics=dict(accuracy=accuracy),
        x=x_train[:1],
        y=y_train[:1])

    generator = updater.mini_batch_generator(
        x_train, y_train, batch_size=16, n_epochs=1)
    for _, x, y in generator:
      my_updater(x, y)
    my_updater.add_loss_and_ema_loss_to_log()
    my_updater.add_metrics_to_log('Test', x_test, y_test)
    self.assertGreater(my_updater.logs[-1]['value'], 0.7)


if __name__ == '__main__':
  absltest.main()
