# PCP-Dreamer: Prospective Cognitive Pruning for Safe World-Model RL

This repository provides a **simplified yet faithful implementation** of **PCP-Dreamer**, a world-model-based reinforcement learning algorithm integrating **Prospective Cognitive Pruning (PCP)** for safety-aware decision making.

The implementation is designed for:

* Conceptual clarity
* CPU-friendly training
* Methodological alignment with Dreamer-style latent dynamics

---

## 📌 Key Features

* **World Model**: RSSM with LayerNorm and orthogonal initialization
* **Actor Training**: Pathwise derivative (Dreamer-style imagination)
* **Safety Module**: Prospective Cognitive Pruning (PCP) as a soft constraint
* **Environment**: Continuous control (Pendulum-v1 by default)
* **Optimized for CPU**: Stable hyperparameters and reduced update ratio

---

## 🧠 Algorithm Overview

The agent learns a latent dynamics model and performs imagination-based policy optimization.
PCP introduces a **risk-aware cognitive pruning mechanism** by evaluating prospective latent trajectories and shaping policy updates away from unsafe regions.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

Training rewards will be plotted and saved as:

```text
reward_curve.png
```

---

## ⚙️ Configuration

All hyperparameters are defined in the `Config` dataclass inside `train.py`, including:

* RSSM dimensions
* Imagination horizon
* PCP risk bounds
* Optimization settings

---

## 📖 Notes

* This code is intended for **research and educational use**
* PCP constraints are implemented as **latent risk shaping**, not hard safety filters
* The implementation favors readability over maximal performance

---

## 📜 Citation

If you use this code in your research, please cite the corresponding paper (to be added).

---

## 📬 Contact

For questions or discussions, feel free to open an issue.
