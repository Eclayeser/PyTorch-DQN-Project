# Reinforcement Learning — Deep Q-Learning from CartPole to Custom Games

> A progression of Deep Q-Network (DQN) implementations, from a standard baseline to two custom-built game environments, with double DQN comparison and PyTorch → ONNX inference benchmarking.

[![Python](https://img.shields.io/badge/python-3.x-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-DQN-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

<p align="center">
  <img src="cartpole-studying/gifs/cartpole-episode-3-gif.gif" width="32%" />
  <img src="first-game-iteration/videos-vanilla-DQN/gifs/bounce-episode-3-gif.gif" width="32%" />
  <img src="second-game-iteration/videos-double-DQN/gifs/zone_controller-episode-3-gif.gif" width="32%" />
</p>

**Me (Yellow) struggling to win against my own agent:**

<p align="center">
  <img src="second-game-iteration/videos-double-DQN/gifs/me-vs-agent.gif" width="75%" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [1. Baseline: DQN on CartPole](#1-baseline-dqn-on-cartpole)
- [2. Bounce Platform](#2-bounce-platform)
- [3. Bounce Platform — Double DQN](#3-bounce-platform--double-dqn)
- [4. Zone Capture](#4-zone-capture)
- [5. Inference Benchmarking: PyTorch vs ONNX](#5-inference-benchmarking-pytorch-vs-onnx)
- [Play Yourself](#play-yourself)
- [Key Learnings](#key-learnings)

---

## Overview

This repository demonstrates my learning curve:

- I started by exploring existing learning scripts with the Gymnasium environment, learning the principles of DQN implementation.
- Following the chosen example, I built a first simple game to solidify my understanding and fix any gaps.
- After a successful implementation, I moved on to create a harder second game that would let me play against my own trained AI.
- To finish, I conducted latency benchmarking for a deeper analysis of how exported models can perform more efficiently.

**Why this project:** I've always wanted to train my own agents and play my custom games against them. It was also my introduction to PyTorch and its core building blocks — tensors and neural networks.

---

## Project Structure

```
.
├── first-game-iteration/       # Custom game #1 (Bounce Platform): vanilla DQN + double DQN
├── second-game-iteration/      # Custom game #2 (ZoneCapture): double DQN + benchmarking
├── cartpole-studying/          # Example DQN script used for studying (CartPole): vanilla DQN
```

---

## 1. Baseline: DQN on CartPole

A standard, pre-built DQN implementation trained on the classic `CartPole-v1` environment. I used it to learn and validate the core DQN algorithm before building custom environments.

Source: `https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html`

- **Purpose:** learning exercise for the DQN implementation / experimenting with code
- **Environment:** OpenAI Gym `CartPole-v1`

<p align="center">
  <img src="cartpole-studying/gifs/cartpole-episode-2-gif.gif" width="49%" />
  <img src="cartpole-studying/gifs/cartpole-episode-3-gif.gif" width="49%" />
</p>

**What I learned:**

- I was always concerned about stability in reinforcement learning. This is the first time I got introduced to replay memory, which randomly samples and decorrelates training batches to greatly stabilize the procedure.
- I also finally understood how to employ two neural networks simultaneously for better yields, like using a separate target network to compute expected state values for added stability.
- Having studied Markov Decision Processes at university, I didn't fully grasp their importance until I started analyzing this script. I've been carefully looking at the limitations of this model to figure out the exact requirements my future custom game will need to meet.
- I experimented with managing the exploration-exploitation trade-off by changing the epsilon start/end/decay constants. The script ensures the agent initially favors random actions to explore the space, but this randomness decays exponentially over time so the agent can eventually exploit its learned policy to maximize rewards. This makes perfect sense: the agent has to see all the different outcomes and possibilities first before it can actually decide which actions are best to take.
- My biggest takeaway was that changing the epsilon decay constant should really be linked to the number of episodes run. If I run more episodes, I should increase the decay constant so there is more time for exploration before sticking to the approach the agent deems best.

---

## 2. Bounce Platform

A simple game I built using the Gymnasium environment: the agent controls a platform that must move left/right to keep a ball from falling past the bottom of the screen. Made it fast-paced on purpose in order to reduce training time.

<p align="center">
  <img src="first-game-iteration/videos-vanilla-DQN/gifs/bounce-episode-1-gif.gif" width="32%" />
  <img src="first-game-iteration/videos-vanilla-DQN/gifs/bounce-episode-2-gif.gif" width="32%" />
  <img src="first-game-iteration/videos-vanilla-DQN/gifs/bounce-episode-3-gif.gif" width="32%" />
</p>

**Progress Graph:**

<p align="center">
  <img src="first-game-iteration/vanilla_DQN_graph.png" width="49%" />
</p>

**Results:**

Unexpectedly, the Double DQN agent learned slightly worse than the vanilla DQN on this environment given the same number of episodes. I ran a couple of runs to verify this behavior, and all of them produced graphs like the ones above:

- In the first graph, the average performance line is seen with a noticeable dip in the middle of training.
- The second run's graph was even more concerning, as the average duration even started going drastically down after episode 500.

I figured this sudden crash in performance is a classic sign of policy collapse, often referred to as catastrophic forgetting or instability. This phenomenon typically happens when the exploration rate epsilon decays too quickly, causing the agent to overfit to a narrow set of experiences stored in its replay buffer. Without continuously gathering new exploratory data, the neural network effectively "forgets" how to properly recover from edge-case states, causing the overall performance to plummet.

**For the future:** if I want to improve the Double DQN's performance and resolve this instability, a primary measure would be independent hyperparameter tuning. Rather than simply reusing the vanilla DQN settings, it would be highly beneficial to conduct a grid search specifically tailored for the DDQN. Focus will be placed heavily on tuning the learning rate and the target network update frequency (or the soft-update parameter, tau, which should help the network evaluate and update its policies much more stably).

---

## 3. Bounce Platform — Double DQN

I decided to explore the concept of Double DQN: it was developed in order to resolve vanilla DQN's tendency to overestimate action values, aka maximisation bias. The same Bounce Platform environment was used.

**Progress Graphs (2 runs):**

<p align="center">
  <img src="first-game-iteration/double_DQN_graph_1.png" width="49%" />
  <img src="first-game-iteration/double_DQN_graph_2.png" width="49%" />
</p>

**Results:**

Unexpectedly, the Double DQN agent learned slightly **worse** than the vanilla DQN on this environment given the same number of episodes.

I run a couple of runs and all of them produced graphs like the ones above:

- First graph's average line is seen with a dip in the middle.
- The second one even started going dratically down after episode 500.

In my view, this drop in performance is a clear example of policy failure, known as catastrophic forgetting or instability, which takes place when the exploration factor epsilon is reduced too fast and the agent becomes overfitted to the experience collected in the replay memory. In such circumstances, without collecting new exploratory data continuously, the network will eventually forget how to behave in the case of edge states and will cause the performance to fall sharply.

**As to the improvements in the future:** if I want to increase the performance of the Double DQN and solve the problem of its instability, one of the most effective ways would be to conduct the independent hyperparameter tuning. By using not only the default DQN settings, but also performing the grid search for the DDQN, the emphasis will be put on adjusting the learning rate and the frequency of updates of the target network (tau).

---

## 4. Zone Capture

A significantly harder environment where the agent plays **against** the user, having been trained against a scripted bot opponent. This was much more fun to implement and test, as I could physically play against my own trained model.

<p align="center">
  <img src="second-game-iteration/videos-double-DQN/gifs/zone_controller-episode-1-gif.gif" width="32%" />
  <img src="second-game-iteration/videos-double-DQN/gifs/zone_controller-episode-2-gif.gif" width="32%" />
  <img src="second-game-iteration/videos-double-DQN/gifs/zone_controller-episode-3-gif.gif" width="32%" />
</p>

**Progress Graph:**

<p align="center">
  <img src="second-game-iteration/graph.png" width="49%" />
</p>

**What made this harder:**

- Number of possible actions risen from 3 to 9
- Number of observations risen from 6 to 14
- The game logic itself is much harder: the agent not only has to figure out that it needs to be in the moving zone to score, but also that it has to compete its opponent out of it + avoid staying inside the zone when it's hot (while still following it nearby for quick re-enter)

**Results:**

It took about 1 hour to run the final training (having done multiple previously).

- Initially, the agent would move randomly.
- It then started figuring out the slight path shape it had to perform.
- Eventually, in the last run, you can clearly see how the agent not only stays within the zone, but also pushes its opponent away in order to claim full zone control as intended.
- It still had not yet fully understood that it had to be out of the zone when it's hot, but this issue was not critical since the agent kept beating the bot every time.

To accommodate the increased complexity, I had to modify the learning script as follows:

- **Training Length** (increased to 1200 episodes): provides the nn the necessary time to experience this vastly larger state space and converge on a winning policy.
- **Exploration Rate** (EPS_DECAY increased to 450,000): the agent needs to try many more combinations of actions to understand their effects.
- **Replay Memory** (capacity increased to 250,000): a small memory of 10,000 transitions would overwrite older, still-valuable experiences too quickly. Higher capacity ensures the agent samples from a rich, diverse history of states, breaking correlations in the training data and stabilizing the learning process.
- **Optimisation Frequency** (optimise on every 4th episode): the agent gathers a bit more distinct experience in the environment between each neural network weight update. This smooths out the learning trajectory and drastically improves the wall-clock execution time of the script.
- **Discount Factor** (increased from 0.99 to 0.995): forces the agent to look further into the future. This is critical for the new game logic because the agent must plan ahead for delayed events, such as anticipating the end of the 3.0-second "hot" duration so it can safely re-enter the zone the moment it turns neutral.

---

## 5. Inference Benchmarking: PyTorch vs ONNX

I conducted a short benchmarking exercise (for the Zone Capture game's best policy) comparing raw PyTorch inference latency against an ONNX-exported version of the same trained model.

**Method:**

- I start off with creating a random input tensor and performing warm-up of 100 repetitions for both formats.
- After this, I measure the total amount of time taken by the system to perform 1,000 repetitions of inference using `time.perf_counter()`.
- I also make sure that gradients are not calculated by PyTorch (with `no_grad` option).

| Backend | Avg. Latency (over 5 runs) |
|---|---|
| PyTorch (raw) | `0.0454 ms` |
| ONNX Runtime | `0.0165 ms` |

By obtaining a 2.75x improvement in speed, I have greatly increased the efficiency of my AI model.

The key aspect to consider is scaling. Since my Deep Q-Network has to make many decisions in a split second in a game, this speedup means that in future, I can significantly increase the tick rate or use many AI agents without overwhelming the system.

Essentially, through this process, I am greatly reducing the computational overhead that is involved, thereby allocating vital CPU power to other processes such as the game logic, physics, or rendering.

---

## Play Yourself

You can play ZoneCapture against the AI agent yourself!

1. Install dependencies from [requirements.txt](requirements.txt)
2. Run:
   ```
   python second-game-iteration/play.py
   ```

If you want to test out other policies, change `model_path` in `second-game-iteration/play.py`, inside the `play_against_agent()` function.

---

## Key Learnings

**Exploring the Right Balance between Exploration and Exploitation**

I have acquired an understanding of how to guide the learning process of an agent over time. I now know how to properly correlate the rate of decay of the exploration factor (epsilon) with the duration of training.

**Reinforcement Learning Algorithms for Large-Scale Environments**

I have learned to modify reinforcement learning algorithms in order to work with larger state and action spaces. I am aware of the ways of tuning key parameters so that complex decision-making would be made possible.

**Engineering Environments for RL Training**

I have experience building custom simulation environments from scratch in order to support training using Deep Q-Learning. I am capable of analyzing weaknesses in a particular model design and engineering specific mechanics, such as calculating reward, controlling the pacing of the simulation or increasing/decreasing observation complexity, to optimize training time.

**Optimization of AI Inference for Production**

I have learnt how to optimize and benchmark trained neural networks to achieve extremely efficient inference performance using ONNX runtime: an important technique for scaling AI without causing bottlenecks on system infrastructure.

---

## Author

Mark Tarnavskyi — [LinkedIn](https://www.linkedin.com/in/marktarnavskyi/)