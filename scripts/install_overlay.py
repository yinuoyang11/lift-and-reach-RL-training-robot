from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_RELATIVE_FILES = [
    "__init__.py",
    "lift_env_cfg.py",
    "mdp/__init__.py",
    "mdp/observations.py",
    "mdp/rewards.py",
    "mdp/terminations.py",
    "config/franka/__init__.py",
    "config/franka/ik_abs_env_cfg.py",
    "config/franka/ik_rel_env_cfg.py",
    "config/franka/joint_pos_env_cfg.py",
    "config/franka/joint_pos_dual_arm_env_cfg.py",
    "config/franka/ik_rel_dual_arm_env_cfg.py",
    "config/franka/agents/__init__.py",
    "config/franka/agents/rsl_rl_ppo_cfg.py",
    "config/franka/agents/rsl_rl_ppo_cnn_cfg.py",
    "config/franka/agents/rl_games_ppo_cfg.yaml",
    "config/franka/agents/sb3_ppo_cfg.yaml",
    "config/franka/agents/skrl_ppo_cfg.yaml",
    "config/franka/agents/robomimic/bc.json",
    "config/franka/agents/robomimic/bcq.json",
]
RSL_RELATIVE_FILES = [
    "modules/__init__.py",
    "modules/actor_critic_vision.py",
    "runners/on_policy_runner.py",
    "utils/__init__.py",
    "utils/utils.py",
    "networks/__init__.py",
    "networks/mlp.py",
    "networks/memory.py",
    "networks/normalization.py",
]


def resolve_task_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    module = importlib.import_module("isaaclab_tasks.manager_based.manipulation.lift")
    return Path(module.__file__).resolve().parent


def resolve_rsl_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    module = importlib.import_module("rsl_rl")
    return Path(module.__file__).resolve().parent


def copy_files(src_root: Path, dst_root: Path, relative_files: list[str]) -> None:
    for rel in relative_files:
        src = src_root / rel
        dst = dst_root / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install lift-and-reach overlay files into IsaacLab and rsl_rl.")
    parser.add_argument("--task-root", help="Target lift task directory inside isaaclab_tasks.")
    parser.add_argument("--rsl-root", help="Target rsl_rl package directory.")
    args = parser.parse_args()

    task_root = resolve_task_root(args.task_root)
    rsl_root = resolve_rsl_root(args.rsl_root)

    copy_files(REPO_ROOT, task_root, TASK_RELATIVE_FILES)
    copy_files(REPO_ROOT / "rsl_rl", rsl_root, RSL_RELATIVE_FILES)

    print("overlay install complete")
