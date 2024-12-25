"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn

# from envs.smac.StarCraft2_Env import actions
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.utils.trpo_util import CCSD, update_model, flat_params
from harl.utils.trpo_util import kl_approx as KL
from harl.utils.multi_divergence import gcsd


class MADPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(MADPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_div = args["use_div"]
        self.div_type = args["div_type"]
        self.div_coef = args["div_coef"]
        self.div_weight = args["div_weight"]
        self.sigma = args["sigma"]
        self.gcsd_sigma = args["gcsd_sigma"]
        self.print_loss = args["print_loss"]

    @staticmethod
    def unzip_sample_data(data):
        if len(data) == 8:
            (
                pre_obs_batch,
                pre_rnn_states_batch,
                pre_actions_batch,
                pre_masks_batch,
                pre_active_masks_batch,
                pre_old_action_log_probs_batch,
                pre_adv_targ,
                pre_available_actions_batch,
                # _,
            ) = data
        else:
            (
                pre_obs_batch,
                pre_rnn_states_batch,
                pre_actions_batch,
                pre_masks_batch,
                pre_active_masks_batch,
                pre_old_action_log_probs_batch,
                pre_adv_targ,
                pre_available_actions_batch,
                _,
            ) = data
        return pre_obs_batch, \
            pre_rnn_states_batch, \
            pre_actions_batch, \
            pre_masks_batch, \
            pre_active_masks_batch, \
            pre_old_action_log_probs_batch, \
            pre_adv_targ, \
            pre_available_actions_batch,

    def update_(self, sample, pre_sample, pre_actor):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            pre_sample:
            pre_actor:
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions_with_input_actor(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            self.actor,
            available_actions_batch,
            active_masks_batch,

        )
        old_action_log_probs, old_dist_entropy, _ = self.evaluate_actions_with_input_actor(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            self.old_actor,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                                         -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                                         * active_masks_batch
                                 ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss
        self.actor_optimizer.zero_grad()

        if not self.use_div:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        else:

            (
                pre_obs_batch,
                pre_rnn_states_batch,
                pre_actions_batch,
                pre_masks_batch,
                pre_active_masks_batch,
                pre_old_action_log_probs_batch,
                _,
                pre_available_actions_batch,
            ) = self.unzip_sample_data(pre_sample)

            pre_action_log_probs, pre_dist_entropy, _ = self.evaluate_actions_with_input_actor(
                pre_obs_batch,
                pre_rnn_states_batch,
                pre_actions_batch,
                pre_masks_batch,
                pre_actor,
                pre_available_actions_batch,
                pre_active_masks_batch,
            )

            old_action_log_probs.detach()
            pre_action_log_probs.detach()

            iter_div = CCSD(obs_batch, obs_batch, action_log_probs, old_action_log_probs, self.device,
                            sigma=self.sigma)
            agent_div = CCSD(obs_batch, pre_obs_batch, action_log_probs, pre_action_log_probs, self.device,
                             sigma=self.sigma)
            div = (1 - self.div_weight) * iter_div + self.div_weight * agent_div

            (policy_loss - div * self.div_coef).backward()

            params = flat_params(self.actor)
            update_model(self.old_actor, params)

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train_(self,
               actor_buffer,
               old_actor_buffer,
               pre_actor_buffer,
               pre_actor,
               advantages,
               state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
            old_actor_buffer
            pre_actor_buffer
            pre_actor
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc.).
        """
        train_info = {"policy_loss": 0, "dist_entropy": 0, "actor_grad_norm": 0, "ratio": 0}

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for i in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
                pre_data_generator = pre_actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
                old_data_generator = old_actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                pre_data_generator = pre_actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                old_data_generator = old_actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                pre_data_generator = pre_actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
                old_data_generator = old_actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )

            for sample, old_sample, pre_sample, pre_pre_sample in zip(data_generator, old_data_generator,
                                                                      pre_data_generator, pre_pre_data_generator):
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update_(
                    sample, pre_sample, pre_pre_sample, pre_actor)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
