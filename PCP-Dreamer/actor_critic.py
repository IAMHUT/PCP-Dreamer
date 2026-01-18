import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from utils import Dense, orthogonal_init
from config import Config

class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, action_limit):
        super().__init__()
        self.action_limit = float(action_limit)
        self.net = nn.Sequential(
            Dense(feature_dim, hidden_dim),
            Dense(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, action_dim * 2)
        )
        self.apply(orthogonal_init)

    def forward(self, feature):
        x = self.net(feature)
        mean, std_raw = torch.chunk(x, 2, dim=-1)
        mean = torch.tanh(mean) * self.action_limit
        std = F.softplus(std_raw) + 0.1
        return td.Independent(td.Normal(mean, std), 1)

    def get_action(self, feature, deterministic=False):
        dist = self.forward(feature)
        if deterministic:
            return dist.mean
        action = dist.rsample()
        action = torch.tanh(action) * self.action_limit
        return action

    def log_prob(self, feature, action):
        dist = self.forward(feature)
        action_clip = torch.clamp(action / self.action_limit, -0.999, 0.999)
        raw = torch.atanh(action_clip)

        lp = dist.log_prob(raw)
        lp = lp - torch.sum(torch.log(1 - action_clip.pow(2) + 1e-6), dim=-1)
        return lp


class Critic(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            Dense(feature_dim, hidden_dim),
            Dense(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, feature):
        return self.net(feature).squeeze(-1)


class PCPModule:
    def __init__(self, cfg, wm, actor, device):
        self.cfg = cfg
        self.wm = wm
        self.actor = actor
        self.device = device

    def _risk(self, state):
        d = torch.norm(state["deter"], dim=-1)
        s = torch.norm(state["stoch"], dim=-1)
        r = self.cfg.pcp_w_deter * F.relu(d - self.cfg.pcp_deter_norm_bound) + \
            self.cfg.pcp_w_stoch * F.relu(s - self.cfg.pcp_stoch_norm_bound)
        return r

    def get_shaping_loss(self, state, current_action_log_prob):
        if not self.cfg.pcp_enabled:
            return 0.0

        with torch.no_grad():
            next_state_single = self.wm.rssm.img_step(state, self.actor.get_action(self.wm.rssm.get_feature(state), True))
            risk_val = self._risk(next_state_single)

        return risk_val * self.cfg.pcp_shaping_scale