from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "Pendulum-v1"
    obs_dim: int = 3
    action_dim: int = 1
    action_limit: float = 2.0

    deter_dim: int = 200
    stoch_dim: int = 30
    hidden_dim: int = 200
    embed_dim: int = 100

    seed: int = 42
    total_steps: int = 100000
    prefill_steps: int = 2000

    batch_size: int = 50
    seq_len: int = 30
    buffer_size: int = 100000

    model_lr: float = 3e-4
    critic_lr: float = 8e-5
    actor_lr: float = 8e-5
    grad_clip: float = 100.0

    kl_scale: float = 0.5
    kl_balance: float = 0.8
    free_nats: float = 3.0

    imagine_horizon: int = 15
    discount: float = 0.99
    lambda_gae: float = 0.95

    actor_entropy_scale: float = 1e-4

    train_every: int = 500
    train_steps: int = 20
    eval_every: int = 5000

    pcp_enabled: bool = True
    pcp_num_candidates: int = 8
    pcp_lookahead: int = 2
    pcp_risk_gamma: float = 0.95
    pcp_tau: float = 1.0
    pcp_shaping_scale: float = 0.5
    pcp_for_critic: bool = False

    pcp_deter_norm_bound: float = 20.0
    pcp_stoch_norm_bound: float = 5.0
    pcp_w_deter: float = 0.1
    pcp_w_stoch: float = 0.1

    plot_dir: str = "plots"
    device: str = "cpu"

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_dim