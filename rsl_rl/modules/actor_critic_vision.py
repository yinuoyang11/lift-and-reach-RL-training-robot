# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDictBase
from torchvision.models import ResNet18_Weights, resnet18

from rsl_rl.networks import MLP


class _VisionBackbone(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"ResNet18 pretrained backbone expects RGB input with 3 channels, got {in_channels}.")

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, out_dim),
            nn.ELU(),
        )

        # Freeze early layers to keep pretrained visual features stable.
        frozen_prefixes = ("0", "1", "4", "5")
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.proj(x)


class ActorCriticVision(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVision.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.image_keys = []
        self.group_feature_specs = {"policy": [], "critic": []}

        num_actor_obs = self._infer_group_features(obs, obs_groups["policy"], "policy")
        num_critic_obs = self._infer_group_features(obs, obs_groups["critic"], "critic")

        if len(self.image_keys) == 0:
            raise ValueError("ActorCriticVision requires at least one image observation in obs_groups['policy'].")

        # build one vision backbone per image key
        self.vision_backbones = nn.ModuleDict()
        self.vision_out_dim = 256
        for key in self.image_keys:
            sample = self._get_nested_obs(obs, key)
            channels = sample.shape[-1] if sample.shape[-1] <= 8 else sample.shape[1]
            self.vision_backbones[key] = _VisionBackbone(in_channels=channels, out_dim=self.vision_out_dim)

        actor_input_dim = num_actor_obs + len(self.image_keys) * self.vision_out_dim
        critic_input_dim = num_critic_obs + len(self.image_keys) * self.vision_out_dim

        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)

        self.actor_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalizer = torch.nn.Identity()

        print(f"Actor Vision MLP: {self.actor}")
        print(f"Critic Vision MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"Expected image tensor with rank 4, got shape: {x.shape}")
        # NHWC -> NCHW
        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return x

    def _infer_group_features(self, obs, group_keys: list[str], set_name: str) -> int:
        vector_dim = 0
        for key in group_keys:
            vector_dim += self._register_feature(obs[key], set_name, prefix=key)
        return vector_dim

    def _register_feature(self, sample, set_name: str, prefix: str) -> int:
        if isinstance(sample, TensorDictBase):
            vector_dim = 0
            for sub_key, sub_sample in sample.items():
                vector_dim += self._register_feature(sub_sample, set_name, f"{prefix}/{sub_key}")
            return vector_dim
        if len(sample.shape) == 4:
            if prefix not in self.image_keys:
                self.image_keys.append(prefix)
            self.group_feature_specs[set_name].append(("image", prefix))
            return 0
        if len(sample.shape) == 2:
            self.group_feature_specs[set_name].append(("vector", prefix))
            return sample.shape[-1]
        raise ValueError(f"Unsupported {set_name} obs shape for key '{prefix}': {sample.shape}")

    def _get_nested_obs(self, obs, key_path: str):
        value = obs
        for key in key_path.split("/"):
            value = value[key]
        return value

    def _collect_obs(self, obs: dict, group_keys: list[str], include_vector: bool) -> torch.Tensor:
        features = []
        set_name = "policy" if group_keys == self.obs_groups["policy"] else "critic"
        for feature_type, key_path in self.group_feature_specs[set_name]:
            x = self._get_nested_obs(obs, key_path)
            if feature_type == "image":
                if key_path not in self.vision_backbones:
                    continue
                x = self._to_nchw(x)
                features.append(self.vision_backbones[key_path](x))
            elif include_vector:
                features.append(x)
        return torch.cat(features, dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs):
        return self._collect_obs(obs, self.obs_groups["policy"], include_vector=True)

    def get_critic_obs(self, obs):
        return self._collect_obs(obs, self.obs_groups["critic"], include_vector=True)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        pass

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
