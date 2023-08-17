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
"""Simple Actor-critic agent.

The agent learns to output a binary action, and expects a single float as
observation.
It can be used to learn most of the tsuite-tasks by calling the
fit_agent_to_tsuite_task function.

The implementation is a simple actor-critic, as described e.g. here
https://arxiv.org/abs/1602.01783.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import chex
import distrax
import dm_env
from dm_env import specs
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
import rlax

from tsuite._src import tsuite
from tsuite._src import updater


_N_DISCRETE_ACTIONS = 2
_ACTION_SPEC = specs.DiscreteArray(
        num_values=_N_DISCRETE_ACTIONS, name='discrete')
_OBSERVATION_NAME_1 = 'float'
_OBSERVATION_NAME_2 = 'float_2'
_OBSERVATION_SPEC = {
    _OBSERVATION_NAME_1: specs.BoundedArray(
        shape=(1,), minimum=-1, maximum=1,
        dtype=np.float32, name=_OBSERVATION_NAME_1),
    _OBSERVATION_NAME_2: specs.BoundedArray(
        shape=(1,), minimum=-1, maximum=1,
        dtype=np.float32, name=_OBSERVATION_NAME_2),
}

_AGENT_DISCOUNT = 0.99
_AGENT_LAMBDA = 0.9
_AGENT_ENTROPY_WEIGHT = 2e-2


expand_dim = rlax.tree_fn(lambda x: jnp.expand_dims(x, axis=0))
squeeze_dim = rlax.tree_fn(lambda x: jnp.squeeze(x, axis=0))


def _get_core() -> hk.RNNCore:
  return hk.ResetCore(hk.LSTM(16))


def _initial_state(batch_size: int) -> chex.ArrayTree:
  _, net_state_init = hk.without_apply_rng(
      hk.transform(lambda bs: _get_core().initial_state(bs)))
  return net_state_init({}, batch_size)


def _network(
    timesteps: dm_env.TimeStep,
    state: chex.ArrayTree,
) -> tuple[distrax.Distribution, chex.Array, chex.ArrayTree]:
  """Returns agent policy, value and state."""
  # Network supports arbitrary observation-specs by:
  # - flattening the observation tree
  # - reshaping all observations to [T, B, C]
  # - converting all observations to float32
  # - concatenating all observations into a single array.
  obs = jax.tree_util.tree_leaves(timesteps.observation)
  obs = [x.reshape(x.shape[:2] + (-1,)) for x in obs]
  obs = [x.astype(jnp.float32) for x in obs]
  obs = jnp.concatenate(obs, axis=-1)
  x = hk.BatchApply(hk.nets.MLP(output_sizes=[8, 8]))(obs)
  x, state = hk.dynamic_unroll(
      _get_core(), (x, timesteps.step_type == dm_env.StepType.FIRST), state)
  x = hk.BatchApply(hk.nets.MLP(output_sizes=[8, _N_DISCRETE_ACTIONS]))(x)
  v = hk.BatchApply(hk.nets.MLP(output_sizes=[8, 1]))(x)
  return distrax.Softmax(x), jnp.squeeze(v, axis=-1), state


def _actor_step(
    timesteps: dm_env.TimeStep,
    state: chex.ArrayTree) -> tuple[chex.Array, chex.ArrayTree]:
  """Compute actions for an entire batch of environments."""
  dist, unused_value, state = _network(expand_dim(timesteps), state)
  return squeeze_dim(dist.sample(seed=hk.next_rng_key())), state


def _batch(trees: Sequence[chex.ArrayTree]) -> chex.ArrayTree:
  return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *trees)


def _mean_return(timesteps: dm_env.TimeStep) -> chex.Array:
  n_episodes = jnp.sum(timesteps.step_type == dm_env.StepType.LAST)
  n_episodes = jnp.maximum(n_episodes, 1)
  return jnp.sum(timesteps.reward) / n_episodes


def _actor_critic_loss(
    timesteps: dm_env.TimeStep,
    actions: chex.Array,
    state: chex.ArrayTree) -> chex.Array:
  """Returns actor-critic loss."""
  # Mask, with 0s only on timesteps of type LAST.
  mask_last = jnp.logical_not(timesteps.step_type == dm_env.StepType.LAST)
  mask_tm1 = mask_last[:-1]
  mask_t = mask_last[1:]
  r_t = timesteps.reward[1:]
  d_t = timesteps.discount[1:] * _AGENT_DISCOUNT * mask_t
  dist, v, unused_state = _network(timesteps, state)
  # Compute lambda return.
  v_tm1 = v[:-1]
  v_t = v[1:]
  batch_lambda_returns = jax.vmap(
      rlax.lambda_returns, in_axes=(1, 1, 1, None), out_axes=1)
  returns = batch_lambda_returns(r_t, d_t, v_t, _AGENT_LAMBDA)
  # Value loss.
  delta_tm1 = jax.lax.stop_gradient(returns) - v_tm1
  v_loss = jnp.mean(mask_tm1 * jnp.square(delta_tm1)) / 2.
  # Policy gradient loss.
  logpia_tm1 = dist.log_prob(actions)[:-1]
  pi_loss = -jnp.mean(mask_tm1 * jax.lax.stop_gradient(delta_tm1) * logpia_tm1)
  # Entropy loss
  entropy_tm1 = dist.entropy()[:-1]
  entropy_loss = -jnp.mean(mask_tm1 * entropy_tm1)
  # Sum and weight losses.
  total_loss = v_loss + pi_loss + _AGENT_ENTROPY_WEIGHT * entropy_loss
  return total_loss


def fit_agent_to_tsuite_task(
    tsuite_task: str,
    batch_size: int = 16,
    unroll_length: int = 16,
    n_updates: int = 100,
    early_stopping_mean_return: Optional[float] = None,
) -> Sequence[Mapping[str, Any]]:
  """Returns logs of training."""
  envs = []
  for _ in range(batch_size):
    envs.append(tsuite.TSuiteEnvironment(
        tsuite_task, _ACTION_SPEC, _OBSERVATION_SPEC, remove_nones=True))

  timesteps_b = _batch([env.reset() for env in envs])
  actions_b = _batch([
      env.read_property(tsuite.PROPERTY_RANDOM_ACTION) for env in envs])
  initial_state_b = _initial_state(batch_size)

  my_updater = updater.Updater(
      optimizer=optax.adam(learning_rate=1e-2),
      loss=_actor_critic_loss,
      rng_key=jax.random.PRNGKey(42),
      metrics=dict(accuracy=_mean_return),
      timesteps=_batch([timesteps_b] * unroll_length),
      actions=_batch([actions_b] * unroll_length),
      state=initial_state_b)
  transformed_actor_step = my_updater.transform(_actor_step, jit=True)

  state_b = initial_state_b
  timesteps_tb = []
  actions_tb = []
  for _ in range(n_updates):
    first_state_b = state_b
    # Reset time-batch buffers, keeping the last item around, creating
    # an overlap of one between subsequent unrolls.
    # On the first iteration the buffers will be empty.
    timesteps_tb = timesteps_tb[-1:]
    actions_tb = actions_tb[-1:]
    while len(timesteps_tb) < unroll_length:
      actions_b, state_b = transformed_actor_step(timesteps_b, state_b)
      timesteps_tb.append(timesteps_b)
      actions_tb.append(actions_b)
      timesteps_b = _batch(
          [envs[i].step(actions_b[i]) for i in range(batch_size)])
    my_updater(_batch(timesteps_tb), _batch(actions_tb), first_state_b)
    my_updater.add_metrics_to_log('Test', _batch(timesteps_tb))
    if early_stopping_mean_return is not None:
      if my_updater.logs[-1]['value'] > early_stopping_mean_return:
        break
  return my_updater.logs
