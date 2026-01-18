import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from utils import Dense, orthogonal_init

class RSSM(nn.Module):
    def __init__(self, action_dim, embed_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.fc_embed_state_action = Dense(stoch_dim + action_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        self.fc_prior = Dense(deter_dim, hidden_dim)
        self.prior_mean = nn.Linear(hidden_dim, stoch_dim)
        self.prior_std = nn.Linear(hidden_dim, stoch_dim)

        self.fc_posterior = Dense(deter_dim + embed_dim, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim, stoch_dim)
        self.posterior_std = nn.Linear(hidden_dim, stoch_dim)

        self.apply(orthogonal_init)
        self._min_std = 0.1

    def init_state(self, batch_size, device):
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def get_feature(self, state):
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def _stats_prior(self, deter):
        h = self.fc_prior(deter)
        mean = self.prior_mean(h)
        std = F.softplus(self.prior_std(h)) + self._min_std
        return {"mean": mean, "std": std}

    def _stats_post(self, deter, embed):
        h = self.fc_posterior(torch.cat([deter, embed], dim=-1))
        mean = self.posterior_mean(h)
        std = F.softplus(self.posterior_std(h)) + self._min_std
        return {"mean": mean, "std": std}

    def obs_step(self, prev_state, prev_action, embed):
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.fc_embed_state_action(x)
        deter = self.gru(x, prev_state["deter"])

        prior = self._stats_prior(deter)
        post = self._stats_post(deter, embed)

        stoch = post["mean"] + post["std"] * torch.randn_like(post["std"])
        return {"deter": deter, "stoch": stoch}, prior, post

    def img_step(self, prev_state, action):
        x = torch.cat([prev_state["stoch"], action], dim=-1)
        x = self.fc_embed_state_action(x)
        deter = self.gru(x, prev_state["deter"])

        prior = self._stats_prior(deter)
        stoch = prior["mean"] + prior["std"] * torch.randn_like(prior["std"])
        return {"deter": deter, "stoch": stoch}

    def observe_sequence(self, embed, action):
        B, Tp1, _ = embed.shape
        device = embed.device
        state = self.init_state(B, device)

        states, priors, posts = defaultdict(list), defaultdict(list), defaultdict(list)

        prev_action = torch.zeros(B, action.shape[-1], device=device)

        for t in range(Tp1):
            state, prior, post = self.obs_step(state, prev_action, embed[:, t])

            if t < Tp1 - 1:
                prev_action = action[:, t]

            for k, v in state.items(): states[k].append(v)
            for k, v in prior.items(): priors[k].append(v)
            for k, v in post.items(): posts[k].append(v)

        return (
            {k: torch.stack(v, 1) for k, v in states.items()},
            {k: torch.stack(v, 1) for k, v in priors.items()},
            {k: torch.stack(v, 1) for k, v in posts.items()}
        )