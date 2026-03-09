# lift-and-reach-RL-training-robot

This repository is an overlay for the working IsaacLab lift task and the matching `rsl_rl` vision policy files.

It is not a standalone replacement for IsaacLab. The intended workflow is:

1. Clone this repository on the target machine.
2. Activate the IsaacLab Python environment.
3. Copy these files into the installed `isaaclab_tasks` and `rsl_rl` packages.
4. Run training from the IsaacLab repository root.

## Included files

Task overlay:
- `lift_env_cfg.py`
- `mdp/observations.py`
- `mdp/rewards.py`
- `mdp/terminations.py`
- `config/franka/__init__.py`
- `config/franka/joint_pos_dual_arm_env_cfg.py`
- `config/franka/ik_rel_dual_arm_env_cfg.py`
- matching Franka agent cfg files

RSL-RL overlay:
- `rsl_rl/modules/actor_critic_vision.py`
- `rsl_rl/runners/on_policy_runner.py`
- minimal `rsl_rl/utils` and `rsl_rl/networks` dependencies used by the vision policy

## Install overlay

If the target machine already has IsaacLab and `rsl_rl` installed in the active Python environment:

```bash
python scripts/install_overlay.py
```

If auto-detection fails, pass explicit package roots:

```bash
python scripts/install_overlay.py \
  --task-root /path/to/site-packages/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift \
  --rsl-root /path/to/site-packages/rsl_rl
```

## Train

After overlay installation, switch to the IsaacLab repository root and run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-DualArm-IK-Rel-v0 \
  --enable_cameras \
  --headless \
  --device cuda:0 \
  --rendering_mode performance
```

Recommended first run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-DualArm-IK-Rel-v0 \
  --enable_cameras \
  --headless \
  --device cuda:0 \
  --rendering_mode performance \
  --num_envs 64
```

## Notes

- The custom task name still says `DualArm`, but the active robot asset is Franka single arm.
- Reward logic is the current working version, including grasp-gated lift rewards.
- The vision backbone is `ResNet18` with `ImageNet-1K` pretrained weights and early layers frozen.
- Existing non-vision checkpoints are not compatible with the vision observation space.
