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
"""A simple updater using JAX ecosystem.

This updater can be used for supervised learning, unsupervised learning and
reinforcement learning. Within tsuite it is used to test the tsuite tasks with
a simple actor-critic agent in an RL setup (see agent.py).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, Union

from absl import logging
import chex
import haiku as hk
import jax
import numpy as np
import optax

T = TypeVar('T')


def _assert_no_jax_tracers(nest):
  def _assert(x):
    msg = 'Detected jax.core.Tracer! This code should not be jit-compiled.'
    assert not isinstance(x, jax.core.Tracer), msg
  jax.tree_map(_assert, nest)


class Updater():
  """Simple Updater using the JAX ecosystem."""

  # pylint: disable=g-bare-generic
  def __init__(
      self,
      optimizer: optax.GradientTransformation,
      loss: Callable[..., Union[tuple, chex.Array]],
      metrics: dict[str, Callable],
      rng_key: chex.Array,
      *loss_args,
      **loss_kwargs):
    (self._update_rng_key,
     self._metric_rng_key,
     self._transformed_rng_key,
     init_rng_key) = jax.random.split(rng_key, 4)
    loss_init, self._loss_apply = hk.transform_with_state(loss)
    self._net_params, self._net_state = jax.jit(loss_init)(
        init_rng_key, *loss_args, **loss_kwargs)
    self._opt_state = optimizer.init(self._net_params)
    self._step_counter = 0
    self._loss = 0
    self._loss_ema = 0
    self._ema_decay = 0.99
    self._metrics = {k: jax.jit(hk.transform_with_state(m).apply)
                     for k, m in metrics.items()}
    self._logs = []

    def _update(
        opt_state: chex.ArrayTree,
        net_params: chex.ArrayTree,
        net_state: chex.ArrayTree,
        rng_key: chex.Array,
        *loss_args,
        **loss_kwargs,
    ) -> tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:

      def loss(net_params, net_state):
        loss_output, net_state = self._loss_apply(
            net_params, net_state, rng_key, *loss_args, **loss_kwargs)
        if isinstance(loss_output, tuple):
          return loss_output[0], (net_state, loss_output)
        return loss_output, (net_state, (loss_output,))
      gradients, (net_state, loss_output) = jax.grad(
          loss, has_aux=True)(net_params, net_state)
      updates, opt_state = optimizer.update(gradients, opt_state)
      net_params = optax.apply_updates(net_params, updates)
      return opt_state, net_params, net_state, loss_output
    self._update = jax.jit(_update)

  def __call__(
      self, *loss_args, **loss_kwargs
  ) -> tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    self._update_rng_key, rng_key = jax.random.split(self._update_rng_key)
    (self._opt_state,
     self._net_params,
     self._net_state,
     loss_output) = self._update(
         self._opt_state, self._net_params, self._net_state, rng_key,
         *loss_args, **loss_kwargs)
    self._step_counter += 1
    self._loss = float(loss_output[0])
    if self._step_counter == 1:
      self._loss_ema = self._loss
    self._loss_ema *= self._ema_decay
    self._loss_ema += (1 - self._ema_decay) * self._loss
    return self._net_params, self._net_state, loss_output

  def add_loss_and_ema_loss_to_log(self) -> None:
    """Returns logs and adds loss and ema loss to the logs of the updater."""
    self._logs.append(dict(
        label='Train',
        name='loss',
        step=self._step_counter,
        value=self._loss))
    self._logs.append(dict(
        label='Train',
        name='ema_loss',
        step=self._step_counter,
        value=self._loss_ema))

  def add_metrics_to_log(
      self, label: str, *args, **kwargs) -> None:
    """Returns metric logs and adds them to the logs of the updater."""
    self._metric_rng_key, *rng_keys = jax.random.split(
        self._metric_rng_key, len(self._metrics) + 1)
    metric_logs = []
    for (name, fn), rng_key in zip(
        self._metrics.items(), rng_keys, strict=True):
      value = fn(self._net_params, self._net_state, rng_key, *args, **kwargs)[0]
      metric = dict(
          label=label, name=name, step=self._step_counter, value=float(value))
      log_msg = f"{metric['label']} {metric['name']} {metric['value']:.3f}"
      logging.info(log_msg)
      metric_logs.append(metric)
    self._logs.extend(metric_logs)

  def transform(
      self, fn: Callable[..., T], jit: bool = False) -> Callable[..., T]:
    """Returns hk.transform_with_state (and optionally jitted) function."""
    apply_fn = hk.transform_with_state(fn).apply
    if jit:
      apply_fn = jax.jit(apply_fn)
    def transformed_fn(*args, **kwargs):
      # Ensure that the user doesn't jit compile the transformed_fn, because
      # otherwise the params, state and rng is fixed!
      # We could have a more general solution here, but for now we just make
      # sure that it does break with an assert instead of silently failing.
      _assert_no_jax_tracers((args, kwargs))
      self._transformed_rng_key, rng_key = jax.random.split(
          self._transformed_rng_key)
      output, unused_net_state = apply_fn(
          self._net_params, self._net_state, rng_key, *args, **kwargs)
      return output
    return transformed_fn

  @property
  def logs(self) -> Sequence[Mapping[str, Any]]:
    return self._logs


def mini_batch_generator(
    dataset: chex.Array,
    *additional_datasets,
    batch_size: int,
    n_epochs: int,
    progress_callback_fn: Callable[[float], Any] = lambda x: None,
    ):
  """Yields mini-batches.

  Usage in a colab:

  .. code-block:: python

    from colabtools.interactive_widgets import ProgressBar
    def mini_batch_generator(*args, **kwargs):
      progress_bar = ProgressBar()
      progress_bar.Publish()
      yield from updater.mini_batch_generator(
          *args, **kwargs, progress_callback_fn=progress_bar.SetProgress)
      progress_bar.Unpublish()

  Usage for supervised learning on images with labels:
  
  .. code-block:: python

    for x, y in updater.mini_batch_generator(images, labels):
      ...

  Args:
    dataset: the dataset to split into mini-batches.
    *additional_datasets: additional datasets to split into mini-batches.
    batch_size: mini-batch size.
    n_epochs: number of epochs.
    progress_callback_fn: called once per mini-batch with a float indicating
      the progress between 0 and 100.
  """
  n_samples = len(dataset)
  indices = np.arange(n_samples)
  total = n_epochs * n_samples // batch_size
  i = 0
  for epoch in range(n_epochs):
    np.random.shuffle(indices)
    for batch in range(0, n_samples, batch_size):
      progress_callback_fn(i / total * 100.0)
      batch_indices = indices[batch:batch+batch_size]
      additional_batch = tuple(d[batch_indices] for d in additional_datasets)
      yield epoch, dataset[batch_indices], *additional_batch
      i += 1
