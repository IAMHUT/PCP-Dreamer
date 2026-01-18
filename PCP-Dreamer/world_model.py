import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from rssm import RSSM
from utils import Dense, orthogonal_init
from config import Config

class WorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

        self.rssm = RSSM(cfg.action_dim, cfg.embed_dim, cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim)

        self.decoder = nn.Sequential(
            Dense(cfg.feature_dim, cfg.hidden_dim),
            Dense(cfg.hidden_dim, cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.obs_dim),
        )

        self.reward_pred = nn.Sequential(
            Dense(cfg.feature_dim, cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.continue_pred = nn.Sequential(
            Dense(cfg.feature_dim, cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.apply(orthogonal_init)

    def _kl_loss(self, post, prior):
        post_dist = td.Normal(post["mean"], post["std"])
        prior_dist = td.Normal(prior["mean"], prior["std"])
        kl = td.kl_divergence(post_dist, prior_dist)
        kl = kl.sum(dim=-1)

        dist_lhs = td.kl_divergence(post_dist, td.Normal(prior["mean"].detach(), prior["std"].detach())).sum(dim=-1)
        dist_rhs = td.kl_divergence(td.Normal(post["mean"].detach(), post["std"].detach()), prior_dist).sum(dim=-1)

        loss = self.cfg.kl_balance * dist_lhs + (1 - self.cfg.kl_balance) * dist_rhs
        loss = torch.maximum(loss, torch.tensor(self.cfg.free_nats, device=loss.device))
        return loss.mean()

    def compute_loss(self, obs, action, reward, terminal):
        embed = self.encoder(obs)
        states, priors, posts = self.rssm.observe_sequence(embed, action)
        feat = self.rssm.get_feature(states)

        recon = self.decoder(feat)
        loss_recon = F.mse_loss(recon, obs)

        feat_pred = feat[:, 1:]
        reward_pred = self.reward_pred(feat_pred).squeeze(-1)
        loss_reward = F.mse_loss(reward_pred, reward)

        cont_pred = self.continue_pred(feat_pred).squeeze(-1)
        loss_cont = F.binary_cross_entropy_with_logits(cont_pred, 1.0 - terminal)

        loss_kl = self._kl_loss(posts, priors)

        total_loss = loss_recon + loss_reward + loss_cont + self.cfg.kl_scale * loss_kl

        return total_loss, {
            "recon": loss_recon.item(),
            "reward": loss_reward.item(),
            "kl": loss_kl.item(),
            "total": total_loss.item()
        }