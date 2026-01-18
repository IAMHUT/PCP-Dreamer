import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from world_model import WorldModel
from actor_critic import Actor, Critic, PCPModule
from config import Config

class DreamerAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.wm = WorldModel(cfg).to(self.device)
        self.actor = Actor(cfg.feature_dim, cfg.action_dim, cfg.hidden_dim, cfg.action_limit).to(self.device)
        self.critic = Critic(cfg.feature_dim, cfg.hidden_dim).to(self.device)
        self.critic_target = Critic(cfg.feature_dim, cfg.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.pcp = PCPModule(cfg, self.wm, self.actor, self.device)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=cfg.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self._state = None
        self._action = None

    def reset(self):
        self._state = self.wm.rssm.init_state(1, self.device)
        self._action = torch.zeros(1, self.cfg.action_dim, device=self.device)

    @torch.no_grad()
    def policy(self, obs, training=True):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        embed = self.wm.encoder(obs)

        self._state, _, _ = self.wm.rssm.obs_step(self._state, self._action, embed)
        feat = self.wm.rssm.get_feature(self._state)

        if training:
            action = self.actor.get_action(feat)
        else:
            action = self.actor.get_action(feat, deterministic=True)

        self._action = action
        return action.cpu().numpy().flatten()

    def update(self, buffer):
        batch = buffer.sample(self.cfg.batch_size, self.cfg.seq_len)
        if not batch: return {}

        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        terminal = batch["terminal"].to(self.device)

        wm_loss, wm_metrics = self.wm.compute_loss(obs, action, reward, terminal)

        self.wm_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
        self.wm_opt.step()

        with torch.no_grad():
            embed = self.wm.encoder(obs)
            states, _, _ = self.wm.rssm.observe_sequence(embed, action)
            start_state = {k: v.detach().reshape(-1, v.shape[-1]) for k, v in states.items()}

        metrics_ac = self._update_actor_critic(start_state)

        return {**wm_metrics, **metrics_ac}

    def _update_actor_critic(self, start_state):
        horizon = self.cfg.imagine_horizon
        state = start_state

        feats = []
        rewards = []
        conts = []
        pcp_costs = []

        for t in range(horizon):
            feat = self.wm.rssm.get_feature(state)
            feats.append(feat)

            action = self.actor.get_action(feat)

            state = self.wm.rssm.img_step(state, action)

            next_feat = self.wm.rssm.get_feature(state)
            r = self.wm.reward_pred(next_feat).squeeze(-1)
            c = self.wm.continue_pred(next_feat).squeeze(-1)

            if self.cfg.pcp_enabled:
                risk = self.pcp._risk(state)
                pcp_costs.append(risk)

            rewards.append(r)
            conts.append(c)

        feats = torch.stack(feats, dim=0)
        rewards = torch.stack(rewards, dim=0)
        conts = torch.stack(conts, dim=0)
        pcont = torch.sigmoid(conts)

        with torch.no_grad():
            target_val = self.critic_target(feats)
            last_val = target_val[-1]

            lambda_returns = []
            ret = last_val
            for t in reversed(range(horizon)):
                ret = rewards[t] + self.cfg.discount * pcont[t] * (
                    (1 - self.cfg.lambda_gae) * target_val[t] + self.cfg.lambda_gae * ret
                )
                lambda_returns.append(ret)
            lambda_returns = torch.stack(lambda_returns[::-1], dim=0)

        pred_val = self.critic(feats.detach())
        critic_loss = F.mse_loss(pred_val, lambda_returns)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(0.02 * p.data + 0.98 * tp.data)

        v_pi = self.critic(feats)

        lambda_returns_grad = []
        ret_grad = v_pi[-1]
        for t in reversed(range(horizon)):
            ret_grad = rewards[t] + self.cfg.discount * pcont[t] * (
                (1 - self.cfg.lambda_gae) * v_pi[t] + self.cfg.lambda_gae * ret_grad
            )
            lambda_returns_grad.append(ret_grad)
        lambda_returns_grad = torch.stack(lambda_returns_grad[::-1], dim=0)

        pcp_loss = 0
        if self.cfg.pcp_enabled:
            pcp_costs = torch.stack(pcp_costs, dim=0)
            pcp_loss = (pcp_costs * self.cfg.pcp_shaping_scale).mean()

        dist = self.actor.forward(feats.detach())
        entropy = dist.base_dist.entropy().mean()

        actor_loss = -lambda_returns_grad.mean() - (self.cfg.actor_entropy_scale * entropy) + pcp_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_ent": entropy.item(),
            "value_mean": lambda_returns.mean().item(),
            "pcp_loss": pcp_loss.item() if isinstance(pcp_loss, torch.Tensor) else 0
        }