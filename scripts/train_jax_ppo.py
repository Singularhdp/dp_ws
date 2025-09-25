# Copyright 2025 DeepMind Technologies Limited
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
"""Train a PPO/... agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import methods
from methods.envs import registry
from methods.envs import wrapper
from methods.configs import locomotion_params
from methods.configs import manipulation_params
import tensorboardX
import wandb


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "G1JoystickFlatTerrain",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_AGENT = flags.DEFINE_string(
    "agent",
    "ppo",
    f"Name of the agent. One of {', '.join(registry.ALL_AGENTS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)


_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)


def get_rl_config(env_name: str, agent_name: str) -> config_dict.ConfigDict:
  if env_name in methods.locomotion._envs:
    if agent_name.lower() == "ppo":
      return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif agent_name.lower() == "fql":
      return locomotion_params.brax_fql_config(env_name, _IMPL.value)
    
    return ValueError("Invalid agent name.")

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )

def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value

  rl_params = get_rl_config(_ENV_NAME.value, _AGENT.value)

  if _NUM_TIMESTEPS.present:
    rl_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    rl_params.num_timesteps = 0
  if _NUM_EVALS.present:
    rl_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    rl_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    rl_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    rl_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    rl_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    rl_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    rl_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    rl_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    rl_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    rl_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    rl_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    rl_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    rl_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    rl_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    rl_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    rl_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    rl_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    rl_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    rl_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    rl_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _RUN_EVALS.present:
    rl_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    rl_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    rl_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  env = registry.load(_ENV_NAME.value, config=env_cfg)

  print(f"Environment Config ({_ENV_NAME.value}):\n{env_cfg}")
  print(f"{_AGENT.value} Training Parameters:\n{rl_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(rl_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks.make_ppo_networks
  )
  if hasattr(rl_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **rl_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = (
      rl_params.get("num_eval_envs", 128)
  )

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn= wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  # Load evaluation environment.
  eval_env = None
  if True:
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  num_envs = 1

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
    rscope_env = wrapper.wrap_for_brax_training(
        rscope_env,
        episode_length=rl_params.episode_length,
        action_repeat=rl_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        rl_params,
        False,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'rollout{i}.mp4'.")


if __name__ == "__main__":
  app.run(main)