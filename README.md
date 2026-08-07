# [Project Name] — Deep Q-Learning from CartPole to Custom Games

> A progression of Deep Q-Network (DQN) implementations, from a standard baseline to two custom-built game environments, with double DQN comparison and PyTorch → ONNX inference benchmarking.

[![Python](https://img.shields.io/badge/python-3.x-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-DQN-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

<!-- Optional: add a hero GIF here from your best video (Zone Controller is probably the strongest visual) -->
<!-- ![demo](path/to/best_clip.gif) -->

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [1. Baseline: DQN on CartPole](#1-baseline-dqn-on-cartpole)
- [2. Ball Bouncer](#2-ball-bouncer)
- [3. Ball Bouncer — Double DQN](#3-ball-bouncer--double-dqn)
- [4. Zone Controller](#4-zone-controller)
- [5. Inference Benchmarking: PyTorch vs ONNX](#5-inference-benchmarking-pytorch-vs-onnx)
- [Key Takeaways](#key-takeaways)
- [Setup & Usage](#setup--usage)
- [Future Work](#future-work)

---

## Overview

[2–4 sentences: what this repo demonstrates end-to-end. E.g. "This repository documents my journey learning reinforcement learning, starting from a standard DQN on CartPole, through two self-built game environments of increasing difficulty, a comparison against Double DQN, and a short benchmarking exercise on deployment-side inference speed."]

**Why this project:** [one or two lines — what you wanted to learn / prove to yourself]

---

## Project Structure

```
.
├── cartpole-dqn/           # Baseline DQN, off-the-shelf environment
├── ball-bouncer/           # Custom game #1 + vanilla DQN
├── ball-bouncer-ddqn/      # Same game, trained with Double DQN
├── zone-controller/        # Custom game #2 (agent vs. player)
├── onnx-benchmark/         # PyTorch vs ONNX latency comparison
└── README.md
```

[Adjust to match your actual folder/file names]

---

## 1. Baseline: DQN on CartPole

A standard, pre-built DQN implementation trained on the classic `CartPole-v1` environment, used to learn and validate the core DQN algorithm before building custom environments.

- **Purpose:** learning exercise / sanity check for the DQN implementation
- **Environment:** OpenAI Gym `CartPole-v1`
- **File(s):** `[filename(s)]`

[Optional: 1 line on what you learned here, e.g. replay buffer, target network, epsilon decay]

---

## 2. Ball Bouncer

A simple custom environment I built from scratch: the agent controls a platform that must move left/right to keep a ball from falling past the bottom of the screen.

- **File(s):** `[training script]`, `[environment/game file]`
- **Videos:** `[video 1 — early episodes]`, `[video 2 — mid training]`, `[video 3 — final performance]`
- **Metrics:** duration (episode length) vs. episode plot — `[graph filename]`

**Results:**
[1–3 sentences: how quickly it converged, what "solved" looked like, anything notable in the graph]

<!-- Embed if hosting videos/gifs in-repo or on YouTube -->
<!-- ![ball bouncer progress](path/to/clip.gif) -->

---

## 3. Ball Bouncer — Double DQN

The same Ball Bouncer environment, this time trained with Double DQN (DDQN) to compare against the vanilla DQN baseline above.

- **File(s):** `[training script]`
- **Videos:** `[video links/paths]`
- **Metrics:** `[graph filenames — e.g. duration vs episode, loss vs episode]`

**Results:**
Counter-intuitively, the Double DQN agent learned **worse** than the vanilla DQN on this environment. [1–2 sentences describing the gap — how much worse, at what point training diverged/plateaued]

**Why this is still a useful result:**
Reinforcement learning is highly non-deterministic, and "more sophisticated" doesn't always mean "better" on a given seed/environment. Some possible explanations I considered:
- [ ] Ball Bouncer's reward signal may already be simple/dense enough that DDQN's overestimation-bias correction has little to fix, while adding variance elsewhere
- [ ] Hyperparameters tuned for vanilla DQN may not transfer directly to DDQN
- [ ] Random seed / replay buffer sampling variance across a single run
- [ ] Small state space means the overestimation bias DDQN corrects for may not have been a real problem in vanilla DQN here

[Fill in / trim this list to whichever explanations you actually believe, and mention if you re-ran with multiple seeds to check variance]

---

## 4. Zone Controller

*(Suggest renaming this section header to your game's actual name)*

A significantly harder custom environment where the agent plays **against** the user, having been trained against a scripted bot opponent. This is the strongest result in the repo — I still struggle to consistently beat the trained agent.

- **File(s):** `[training script]`, `[environment/game file]`, `[bot opponent script]`
- **Videos:** `[progress videos — early to final]`
- **Metrics:** score vs. episode — `[graph filename]`

**What made this harder:**
[e.g. sparser/adversarial reward, larger state/action space, opponent modeling, longer training time — describe the specific script/architecture adjustments you had to make vs. Ball Bouncer]

**Results:**
[Training time/steps, final performance, how it compares to the bot it trained against, and your own experience playing against it]

<!-- ![zone controller demo](path/to/best_clip.gif) -->

---

## 5. Inference Benchmarking: PyTorch vs ONNX

A short benchmarking exercise comparing raw PyTorch inference latency against an ONNX-exported version of the same trained model.

- **File(s):** `[benchmark script]`
- **Method:** [briefly — number of runs, batch size 1, CPU/GPU, warm-up runs excluded, etc.]

| Backend | Avg. Latency | Notes |
|---|---|---|
| PyTorch (raw) | `[value]` | |
| ONNX Runtime | `[value]` | |

**Result:** ONNX gave a **[X%]** latency improvement over raw PyTorch inference. [1 sentence on why this matters for deployment, even in a toy project — e.g. relevant to real-time inference in trading/production systems]

---

## Key Takeaways

- [What you learned about DQN mechanics: replay buffers, target networks, epsilon-greedy decay, reward shaping]
- [What you learned building environments from scratch vs. using Gym]
- [What the DDQN result taught you about RL variance / evaluation rigor]
- [What the benchmarking taught you about deployment considerations]

---

## Setup & Usage

```bash
# Clone the repo
git clone [repo URL]
cd [repo name]

# Install dependencies
pip install -r requirements.txt

# Train an agent, e.g.
python [ball-bouncer/train.py] --episodes [N]

# Run the ONNX benchmark
python [onnx-benchmark/benchmark.py]
```

**Dependencies:** [PyTorch version, Gym, ONNX Runtime, any custom game engine/rendering library]

---

## Future Work

- [ ] [e.g. re-run Double DQN across multiple seeds to confirm the result isn't noise]
- [ ] [e.g. try Dueling DQN / Prioritized Experience Replay on Zone Controller]
- [ ] [e.g. deploy Zone Controller agent as a playable web demo]

---

## Author

Mark Tarnavskyi — https://www.linkedin.com/in/marktarnavskyi/