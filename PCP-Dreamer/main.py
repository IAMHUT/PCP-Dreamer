import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import Config
from agent import DreamerAgent
from replay_buffer import ReplayBuffer

def main():
    cfg = Config()

    if torch.cuda.is_available():
        cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        cfg.device = "mps"
    print(f"Running on {cfg.device}")

    env = gym.make(cfg.env_name)
    cfg.obs_dim = env.observation_space.shape[0]
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_limit = float(env.action_space.high[0])

    agent = DreamerAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    print("Prefilling buffer...")
    obs, _ = env.reset(seed=cfg.seed)
    agent.reset()
    for _ in range(cfg.prefill_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated or truncated, terminated)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    print("Start Training...")
    global_step = 0
    obs, _ = env.reset(seed=cfg.seed)
    agent.reset()

    ep_rewards = []
    curr_ep_reward = 0

    while global_step < cfg.total_steps:
        action = agent.policy(obs, training=True)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        buffer.add(obs, action, reward, next_obs, done, term)
        curr_ep_reward += reward
        global_step += 1

        if done:
            obs, _ = env.reset()
            agent.reset()
            ep_rewards.append(curr_ep_reward)
            if len(ep_rewards) % 5 == 0:
                avg = np.mean(ep_rewards[-5:])
                print(f"Step {global_step} | EpReward: {curr_ep_reward:.1f} | Avg(5): {avg:.1f}")
            curr_ep_reward = 0
        else:
            obs = next_obs

        if global_step % cfg.train_every == 0:
            metrics_avg = defaultdict(float)
            for _ in range(cfg.train_steps):
                m = agent.update(buffer)
                for k, v in m.items(): metrics_avg[k] += v

            log_str = f"Update @ {global_step} | "
            for k in metrics_avg:
                log_str += f"{k}: {metrics_avg[k]/cfg.train_steps:.3f} | "
            print(log_str)

    plt.plot(ep_rewards)
    plt.title("Training Rewards")
    plt.savefig("reward_curve.png")
    print("Done.")

if __name__ == "__main__":
    main()