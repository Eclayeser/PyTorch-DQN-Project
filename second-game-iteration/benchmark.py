import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(14, 9)

# my best policy
model_path = "second-game-iteration/policies/policy_net_ep_1200.pth"
policy_net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
policy_net.eval()

# random input matching the state space
random_input = torch.randn(1, 14, dtype=torch.float32)

# export PyTorch model to ONNX
torch.onnx.export(policy_net, random_input, "second-game-iteration/policy_net.onnx", 
                  input_names=['state'], output_names=['q_values'])

# benchmark PyTorch
print("Benchmarking Raw PyTorch...")
with torch.no_grad():
    for _ in range(100): # warm up
        policy_net(random_input)
    
    start_time = time.perf_counter()
    for _ in range(1000):
        policy_net(random_input)
    pytorch_time = (time.perf_counter() - start_time) / 1000

# benchmark ONNX runtime
print("Benchmarking ONNX Runtime...")
ort_session = ort.InferenceSession("second-game-iteration/policy_net.onnx")
random_input_np = random_input.numpy()

for _ in range(100): # warm up
    ort_session.run(None, {'state': random_input_np})

start_time = time.perf_counter()
for _ in range(1000):
    ort_session.run(None, {'state': random_input_np})
onnx_time = (time.perf_counter() - start_time) / 1000


print(f"PyTorch Avg Latency: {pytorch_time * 1000:.4f} ms")
print(f"ONNX Avg Latency: {onnx_time * 1000:.4f} ms")
speedup = pytorch_time / onnx_time
print(f"Speedup: ONNX is {speedup:.2f}x faster")
