
# create a game object
# public methods: accept/set action, update game state, return observations of the state

# # Randomness control
# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)

# ReplayMemory is essential so NN - adjust weights using a batch of many transitions (not only one)

# # On GAMMA
# γ=0.99 means a reward 10 steps away is still worth `0.99^10 ≈ 0.90` of its face value — barely discounted,
# so the agent is trained to care almost as much about long-term survival as the very next step. γ=0 would make
# it totally short-sighted (only the immediate reward matters); γ=1 would (in an ongoing task) never converge, since#
# infinite future reward never gets discounted away. 0.99 is a
# standard "care a lot about the long run, but not literally infinitely" choice