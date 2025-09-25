# DP_WS

基于**Mujoco Playground**框架

暂时只保留G1相关环境模型，删除了`_VISION`相关参数内容。



test:

```bash
cd
source mpg/bin/activate
cd dp_ws/
export PYTHONPATH=/home/dp/dp_ws:$PYTHONPATH
python scripts/train_jax_ppo.py --env_name G1JoystickFlatTerrain --rscope_envs 16 --run_evals=False --deterministic_rscope=True
```

