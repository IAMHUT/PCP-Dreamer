import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env_name = "Pendulum-v1"
    state_dim = 3
    action_dim = 1
    action_high = 2.0
    action_low = -2.0


    episodes = 300
    max_steps = 200
    batch_size = 128
    buffer_size = 50000
    gamma = 0.99
    tau = 0.005
    lr = 3e-4

    # PCP参数
    pcp_horizon = 5
    pcp_candidates = 16
    pcp_temperature = 0.5
    risk_threshold = 0.5
    pcp_weight = 0.3
    pcp_start_episode = 30
    pcp_update_freq = 2

    # 网络
    hidden_dim = 128

cfg = Config()


class FastPendulumDynamics:
    """快速向量化的钟摆动力学"""
    def __init__(self):
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        self.max_speed = 8.0

    def step_batch(self, states, actions):
        """
        批量单步推演 (向量化)
        states: [batch, 3]
        actions: [batch, 1]
        """
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = np.arctan2(sin_th, cos_th)

        u = np.clip(actions[:, 0], cfg.action_low, cfg.action_high)


        newthdot = thdot + (-3*self.g/(2*self.l)*np.sin(th) +
                            3/(self.m*self.l**2)*u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        new_states = np.stack([np.cos(newth), np.sin(newth), newthdot], axis=1)
        return new_states.astype(np.float32)

    def rollout_fast(self, states, actions, horizon=cfg.pcp_horizon):
        """
        快速批量推演 (完全向量化)
        states: [batch, state_dim]
        actions: [batch, action_dim]
        返回: 最终风险 (不存储轨迹)
        """
        batch_size = len(states)
        current_states = states.copy()
        total_risk = np.zeros(batch_size)


        current_states = self.step_batch(current_states, actions)
        total_risk += self.compute_risk_batch(current_states)


        zero_actions = np.zeros((batch_size, 1))
        for h in range(1, horizon):
            current_states = self.step_batch(current_states, zero_actions)
            discount = cfg.gamma ** h
            total_risk += self.compute_risk_batch(current_states) * discount

        return total_risk

    @staticmethod
    def compute_risk_batch(states):
        """批量风险计算"""
        th = np.arctan2(states[:, 1], states[:, 0])
        thdot = states[:, 2]

        angle_risk = np.maximum(0.0, np.abs(th) - cfg.risk_threshold)
        speed_risk = np.maximum(0.0, np.abs(thdot) - 5.0) * 0.1

        return angle_risk + speed_risk


class FastActor(nn.Module):
    """轻量级Actor"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.mean = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.log_std = nn.Linear(cfg.hidden_dim, cfg.action_dim)


        nn.init.orthogonal_(self.mean.weight, 0.01)
        self.log_std.weight.data.fill_(0.0)
        self.log_std.bias.data.fill_(-0.5)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * cfg.action_high
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, torch.exp(log_std)

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, cfg.action_low, cfg.action_high)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

class FastCritic(nn.Module):
    """轻量级Critic"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim + cfg.action_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.q = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class FastPCPModule:
    """优化的PCP模块"""
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.update_counter = 0

    def should_update(self):
        """控制PCP更新频率"""
        self.update_counter += 1
        return self.update_counter % cfg.pcp_update_freq == 0

    def compute_pcp_loss_fast(self, actor, states):
        """
        快速PCP损失计算
        只在部分训练步骤中使用
        """
        batch_size = states.shape[0]


        with torch.no_grad():
            mean, std = actor(states)
            dist = Normal(mean, std)


            candidates = dist.sample((cfg.pcp_candidates,))  # [n_cand, batch, act_dim]
            candidates = torch.clamp(candidates, cfg.action_low, cfg.action_high)
            log_probs = dist.log_prob(candidates).sum(-1)  # [n_cand, batch]


        states_np = states.cpu().numpy()
        candidates_np = candidates.permute(1, 0, 2).cpu().numpy()  # [batch, n_cand, act_dim]


        states_flat = np.repeat(states_np, cfg.pcp_candidates, axis=0)  # [batch*n_cand, state_dim]
        actions_flat = candidates_np.reshape(-1, cfg.action_dim)  # [batch*n_cand, act_dim]


        risks_flat = self.dynamics.rollout_fast(states_flat, actions_flat)
        risks = torch.tensor(risks_flat.reshape(batch_size, cfg.pcp_candidates),
                             device=cfg.device, dtype=torch.float32)


        feasibility = F.softmax(-risks / cfg.pcp_temperature, dim=1)


        prior = F.softmax(log_probs.T, dim=1)  # [batch, n_cand]


        posterior = prior * feasibility
        posterior = posterior / (posterior.sum(1, keepdim=True) + 1e-8)

        kl_loss = (posterior * (torch.log(posterior + 1e-8) -
                                torch.log(prior + 1e-8))).sum(1).mean()

        return kl_loss, risks.mean().item()


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=cfg.buffer_size)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = [self.buffer[i] for i in
                 np.random.choice(len(self.buffer), batch_size, replace=False)]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states).to(cfg.device),
            torch.FloatTensor(actions).to(cfg.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(cfg.device),
            torch.FloatTensor(next_states).to(cfg.device),
            torch.FloatTensor(dones).unsqueeze(1).to(cfg.device)
        )

    def __len__(self):
        return len(self.buffer)


class FastPCPAgent:
    """加速的PCP智能体"""
    def __init__(self, use_pcp=True):
        self.use_pcp = use_pcp
        self.dynamics = FastPendulumDynamics()
        self.pcp_module = FastPCPModule(self.dynamics)


        self.actor = FastActor().to(cfg.device)
        self.critic = FastCritic().to(cfg.device)
        self.critic_target = FastCritic().to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.replay_buffer = ReplayBuffer()
        self.pcp_active = False

    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.clamp(mean, cfg.action_low, cfg.action_high)
            else:
                action, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy().flatten()

    def update(self):
        """单次更新"""
        if len(self.replay_buffer) < cfg.batch_size:
            return None, None, None

        batch = self.replay_buffer.sample(cfg.batch_size)
        if batch is None:
            return None, None, None

        states, actions, rewards, next_states, dones = batch


        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * cfg.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        sampled_actions, _ = self.actor.sample(states)
        q_values = self.critic(states, sampled_actions)
        policy_loss = -q_values.mean()


        if self.use_pcp and self.pcp_active and self.pcp_module.should_update():
            kl_loss, avg_risk = self.pcp_module.compute_pcp_loss_fast(self.actor, states)
            total_loss = policy_loss + cfg.pcp_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)
            avg_risk = 0.0
            total_loss = policy_loss

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()


        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        return critic_loss.item(), policy_loss.item(), kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss


def train_fast(use_pcp=True, name="Agent"):
    """快速训练"""
    env = gym.make(cfg.env_name)
    agent = FastPCPAgent(use_pcp=use_pcp)

    returns = []
    risks = []

    pbar = tqdm(range(cfg.episodes), desc=f"Training {name}")

    for episode in pbar:
        state, _ = env.reset()
        episode_return = 0.0
        states_visited = []


        if episode >= cfg.pcp_start_episode and use_pcp:
            agent.pcp_active = True

        for step in range(cfg.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            states_visited.append(state)

            state = next_state
            episode_return += reward


            if len(agent.replay_buffer) >= cfg.batch_size:
                agent.update()

            if done:
                break


        avg_risk = np.mean([agent.dynamics.compute_risk_batch(
            np.array([s]))[0] for s in states_visited])
        returns.append(episode_return)
        risks.append(avg_risk)

        pbar.set_postfix({
            'Ret': f'{episode_return:.0f}',
            'Risk': f'{avg_risk:.3f}',
            'PCP': 'ON' if agent.pcp_active else 'OFF'
        })

    env.close()
    return agent, {'returns': returns, 'risks': risks}


def plot_fast(pcp_stats, baseline_stats):
    """快速绘图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))


    axes[0].plot(pcp_stats['returns'], label='PCP-SAC', alpha=0.7)
    axes[0].plot(baseline_stats['returns'], label='Baseline', alpha=0.7)
    axes[0].set_title("Returns")
    axes[0].legend()
    axes[0].grid(alpha=0.3)


    window = 20
    pcp_smooth = np.convolve(pcp_stats['returns'], np.ones(window)/window, mode='valid')
    base_smooth = np.convolve(baseline_stats['returns'], np.ones(window)/window, mode='valid')
    axes[1].plot(pcp_smooth, label='PCP-SAC', linewidth=2)
    axes[1].plot(base_smooth, label='Baseline', linewidth=2)
    axes[1].set_title("Smoothed Returns")
    axes[1].legend()
    axes[1].grid(alpha=0.3)


    axes[2].plot(pcp_stats['risks'], label='PCP-SAC', alpha=0.7)
    axes[2].plot(baseline_stats['risks'], label='Baseline', alpha=0.7)
    axes[2].axhline(cfg.risk_threshold, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title("Risk")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fast_pcp_results.png', dpi=100)
    plt.show()


    print("\n" + "="*50)
    print("Final Performance (Last 30 episodes):")
    print("="*50)
    print(f"PCP-SAC  | Return: {np.mean(pcp_stats['returns'][-30:]):.1f} | "
          f"Risk: {np.mean(pcp_stats['risks'][-30:]):.3f}")
    print(f"Baseline | Return: {np.mean(baseline_stats['returns'][-30:]):.1f} | "
          f"Risk: {np.mean(baseline_stats['risks'][-30:]):.3f}")
    print("="*50)


def main():
    print("\n🚀 Fast PCP-SAC Experiment\n")

    print("1️⃣ Training PCP-SAC...")
    pcp_agent, pcp_stats = train_fast(use_pcp=True, name="PCP-SAC")

    print("\n2️⃣ Training Baseline...")
    baseline_agent, baseline_stats = train_fast(use_pcp=False, name="Baseline")

    print("\n3️⃣ Plotting...")
    plot_fast(pcp_stats, baseline_stats)

    print("\n✅ Done!")
    return pcp_agent, baseline_agent

if __name__ == "__main__":
    pcp_agent, baseline_agent = main()
