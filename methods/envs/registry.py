"""Registry for all environments."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import ml_collections
from mujoco import mjx

from methods.envs import locomotion
from methods.envs import manipulation
from methods.envs import mjx_env

DomainRandomizer = Optional[
    Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]
]

# A tuple containing all available environment names across all suites.
ALL_ENVS = (
    locomotion.ALL_ENVS # + manipulation.ALL_ENVS
)

# A tuple containing all available agent names.
ALL_AGENTS = (
    "PPO" + "FQL"
)

def get_default_config(env_name: str):
  if env_name in locomotion.ALL_ENVS:
    return locomotion.get_default_config(env_name)

  raise ValueError(f"Env '{env_name}' not found in default configs.")

def load(
    env_name: str,
    config: Optional[ml_collections.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
  if env_name in locomotion.ALL_ENVS:
    return locomotion.load(env_name, config, config_overrides)

  raise ValueError(f"Env '{env_name}' not found. Available envs: {ALL_ENVS}")

def get_domain_randomizer(env_name: str) -> Optional[DomainRandomizer]:
  if env_name in locomotion.ALL_ENVS:
    return locomotion.get_domain_randomizer(env_name)

  return None