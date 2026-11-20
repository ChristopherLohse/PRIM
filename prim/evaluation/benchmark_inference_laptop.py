"""Quick inference time benchmark for the large PRIM model (CPU and MPS if available).

Uses the same data setup as prim/plots/benchmark_inference_time.py:
  NODE_SIZES = [20, 30, 50, 80, 100], n_obs=100, n_int=20, 100 episodes per size.
"""
import time
import numpy as np
import torch
from prim.model.prim_neural_process import PRIMNeuralProcess

CHECKPOINT = "Finalmodels/FinalPRIML.pt"
NODE_SIZES = [20, 30, 50, 80, 100]
N_OBS = 100
N_INT = 20
N_EPISODES = 100
MAX_NODES = 100
_CLIP = 10.0


def _random_obs_int(n_nodes, device, rng):
    obs_raw = rng.standard_normal((N_OBS, n_nodes)).astype(np.float32)
    int_raw = rng.standard_normal((N_INT, n_nodes)).astype(np.float32)
    obs_mean = obs_raw.mean(axis=0, keepdims=True)
    obs_std = obs_raw.std(axis=0, keepdims=True) + 1e-8
    scale = MAX_NODES / n_nodes
    obs_n = np.clip((obs_raw - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    int_n = np.clip((int_raw - obs_mean) / obs_std, -_CLIP, _CLIP) * scale
    obs_pad = np.zeros((N_OBS, MAX_NODES), dtype=np.float32)
    int_pad = np.zeros((N_INT, MAX_NODES), dtype=np.float32)
    obs_pad[:, :n_nodes] = obs_n
    int_pad[:, :n_nodes] = int_n
    obs_t = torch.from_numpy(obs_pad).unsqueeze(0).to(device)
    int_t = torch.from_numpy(int_pad).unsqueeze(0).to(device)
    return obs_t, int_t


def _sync(device_str):
    if device_str == "cuda":
        torch.cuda.synchronize()
    elif device_str == "mps":
        torch.mps.synchronize()


state = torch.load(CHECKPOINT, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

devices = ["cpu"]
if torch.backends.mps.is_available():
    devices.append("mps")

rng = np.random.default_rng(42)

for device_str in devices:
    device = torch.device(device_str)
    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=MAX_NODES,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Warmup
    for n_nodes in [MAX_NODES // 2, MAX_NODES]:
        obs_t, int_t = _random_obs_int(n_nodes, device, rng)
        mask_t = torch.zeros(1, MAX_NODES, device=device)
        mask_t[:, 0] = 1.0
        with torch.no_grad():
            model(obs_t, int_t, target_mask=mask_t, n_nodes=n_nodes)
    _sync(device_str)

    print(
        f"\nDevice: {device_str.upper()}  |  Params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"{'n_nodes':>8}  {'mean_ms':>9}  {'std_ms':>8}  {'ci_lo':>8}  {'ci_hi':>8}"
    )
    print("-" * 52)

    with torch.no_grad():
        for n_nodes in NODE_SIZES:
            mask_t = torch.zeros(1, MAX_NODES, device=device)
            mask_t[:, 0] = 1.0
            times = []
            for _ in range(N_EPISODES):
                obs_t, int_t = _random_obs_int(n_nodes, device, rng)
                _sync(device_str)
                t0 = time.perf_counter()
                logits = model(obs_t,
                               int_t,
                               target_mask=mask_t,
                               n_nodes=n_nodes)
                pad_mask = torch.zeros_like(logits)
                if n_nodes < MAX_NODES:
                    pad_mask[:, n_nodes:] = float("-inf")
                _ = torch.softmax(logits + pad_mask, dim=-1).cpu().numpy()
                _sync(device_str)
                times.append((time.perf_counter() - t0) * 1000)

            arr = np.array(times)
            boot = np.random.default_rng(0).choice(arr,
                                                   size=(500, len(arr)),
                                                   replace=True).mean(axis=1)
            lo, hi = float(np.percentile(boot,
                                         5)), float(np.percentile(boot, 95))
            print(
                f"{n_nodes:>8}  {arr.mean():>8.1f}ms  {arr.std():>7.1f}ms  {lo:>7.1f}ms  {hi:>7.1f}ms"
            )
