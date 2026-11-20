"""
Simplified DAG Data Generation with Strong Soft Interventions

Focuses on edge weight changes for interventions to create clear distribution shifts.
"""

import numpy as np
import networkx as nx
import random
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats

_DEFAULT_NN_CONFIG = Path(__file__).parent / "nn_prior_config.json"

# =============================================================================
# Noise Functions
# =============================================================================

NOISE_TYPES = [
    "gaussian", "poisson", "salt_and_pepper", "truncated_exponential"
]


def _sample_noise(noise_type: str, noise_scale: float, size) -> np.ndarray:
    """Sample zero-centered noise from the specified distribution.

    Args:
        noise_type: One of 'gaussian', 'poisson', 'salt_and_pepper',
                    'truncated_exponential'.
        noise_scale: Controls the spread (analogous to std for Gaussian).
        size: Shape of the output array (int or tuple).

    Returns:
        noise: Zero-centered noise array of the given size.
    """
    if noise_type == "gaussian":
        return np.random.normal(0, noise_scale, size)

    elif noise_type == "poisson":
        # Use lam so that std ~ noise_scale: lam = noise_scale^2 * 4
        lam = max(noise_scale * 4.0, 1.0)
        raw = np.random.poisson(lam, size).astype(float)
        # Center and rescale so spread matches noise_scale
        return (raw - lam) / np.sqrt(lam) * noise_scale

    elif noise_type == "salt_and_pepper":
        # Sparse impulse noise: 10% chance of a large spike, otherwise zero
        spike_prob = 0.1
        noise = np.zeros(size)
        mask = np.random.random(size) < spike_prob
        signs = np.random.choice([-1, 1], size=size)
        noise[mask] = signs[mask] * noise_scale * 5.0
        return noise

    elif noise_type == "truncated_exponential":
        # Exponential truncated at b * scale, then zero-centered
        b = 3.0
        raw = stats.truncexpon.rvs(b=b, scale=noise_scale, size=size)
        mean = stats.truncexpon.mean(b=b, scale=noise_scale)
        return raw - mean

    else:
        raise ValueError(f"Unknown noise type: {noise_type!r}. "
                         f"Choose from {NOISE_TYPES}")


# =============================================================================
# Structural Equation Classes
# =============================================================================


class StructuralEquation:
    """Base class for structural equations."""

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        raise NotImplementedError

    def copy(self) -> "StructuralEquation":
        raise NotImplementedError


class LinearEquation(StructuralEquation):
    """Linear structural equation: Y = sum(w_i * X_i) + noise

    Aligned with CSNP: weights drawn from N(0, 3) and a per-equation
    latent_weight for an injected N(0,1) confounder (handled in sample()).
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        # Random weight for the latent confounder injected in sample() (CSNP-style)
        self.latent_weight = np.random.normal(0, 1.0)

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        if len(parents) == 0:
            return noise
        if self.weights is None:
            # CSNP uses N(0, 10); we use N(0, 3) — wider than U(0.5,2)×sign
            self.weights = np.random.normal(0, 3.0, size=len(parents))
        return np.dot(self.weights, parents) + noise

    def copy(self) -> "LinearEquation":
        eq = LinearEquation(
            self.weights.copy() if self.weights is not None else None)
        eq.latent_weight = self.latent_weight
        return eq


class ConstantBaselineEquation(StructuralEquation):
    """
    Constant-baseline structural equation mimicking availability-style data.

    Normal behavior: Y = baseline + small_noise (nearly constant)
    The intervention effect propagates through parents like normal SCMs,
    but the baseline is high and noise is tiny, so unaffected nodes
    stay near-constant while affected nodes show clear deviations.
    """

    def __init__(self,
                 weights: Optional[np.ndarray] = None,
                 baseline: float = 100.0,
                 noise_scale: float = 0.1):
        self.weights = weights
        self.baseline = baseline
        self.noise_scale = noise_scale

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        if len(parents) == 0:
            return self.baseline + noise * self.noise_scale
        if self.weights is None:
            # Small weights so parent deviations propagate but don't dominate
            self.weights = np.random.uniform(0.1, 0.5, size=len(parents))
            self.weights *= np.random.choice([-1, 1], size=len(parents))
        # Parents are relative to their own baselines, so deviations propagate
        parent_deviations = parents - self.baseline
        return self.baseline + np.dot(
            self.weights, parent_deviations) + noise * self.noise_scale

    def copy(self) -> "ConstantBaselineEquation":
        return ConstantBaselineEquation(
            self.weights.copy() if self.weights is not None else None,
            self.baseline, self.noise_scale)


class NonlinearEquation(StructuralEquation):
    """Nonlinear structural equation: Y = sum(w_i * tanh(s_i * X_i)) + noise"""

    def __init__(self,
                 weights: Optional[np.ndarray] = None,
                 scales: Optional[np.ndarray] = None):
        self.weights = weights
        self.scales = scales
        # Random weight for the latent confounder injected in sample() (CSNP-style)
        self.latent_weight = np.random.normal(0, 1.0)

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        if len(parents) == 0:
            return noise
        if self.weights is None:
            self.weights = np.random.normal(0, 3.0, size=len(parents))
        if self.scales is None:
            self.scales = np.random.uniform(0.8, 1.5, size=len(parents))
        return np.dot(self.weights, np.tanh(self.scales * parents)) + noise

    def copy(self) -> "NonlinearEquation":
        eq = NonlinearEquation(
            self.weights.copy() if self.weights is not None else None,
            self.scales.copy() if self.scales is not None else None,
        )
        eq.latent_weight = self.latent_weight
        return eq


# Torch activation function lookup
TORCH_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": F.relu,
}


def load_nn_config(config_path=None) -> dict:
    """Load neural network prior configuration from JSON file."""
    if config_path is None:
        config_path = _DEFAULT_NN_CONFIG
    with open(config_path, "r") as f:
        return json.load(f)


class NeuralNetEquation(StructuralEquation):
    """
    Neural network structural equation using PyTorch.

    For non-root nodes: Y = NN(parent_values) + noise
    For root nodes: Y = NN(noise_input) where noise_input is the noise value

    The NN architecture is defined by a JSON config file.
    Weights are stored as torch tensors for fast vectorized computation.
    """

    def __init__(
        self,
        nn_weights: Optional[List[torch.Tensor]] = None,
        nn_biases: Optional[List[torch.Tensor]] = None,
        activations: Optional[List[str]] = None,
        layer_units: Optional[List[int]] = None,
        config: Optional[dict] = None,
        n_parents: int = 0,
    ):
        self.nn_weights = nn_weights
        self.nn_biases = nn_biases
        self.activations = activations
        self.layer_units = layer_units
        self.config = config or load_nn_config()
        self.n_parents = n_parents
        self._initialized = nn_weights is not None

    def _initialize(self, input_dim: int):
        """Initialize NN weights as torch tensors based on config and input dimension."""
        config = self.config
        possible_activations = config["possible_activations"]
        layers = config["layers"]
        w_low = config["weight_init"]["low"]
        w_high = config["weight_init"]["high"]
        b_low = config["bias_init"]["low"]
        b_high = config["bias_init"]["high"]

        self.nn_weights = []
        self.nn_biases = []
        self.activations = []
        self.layer_units = []

        prev_dim = input_dim
        for layer_cfg in layers:
            units = layer_cfg["units"]
            self.layer_units.append(units)

            # Initialize weights and biases as torch tensors
            w_np = np.random.uniform(w_low, w_high, size=(prev_dim, units))
            b_np = np.random.uniform(b_low, b_high, size=(units, ))
            self.nn_weights.append(torch.from_numpy(w_np).float())
            self.nn_biases.append(torch.from_numpy(b_np).float())

            # Pick activation
            act = layer_cfg["activation"]
            if act == "random":
                act = np.random.choice(possible_activations)
            self.activations.append(act)

            prev_dim = units

        self._initialized = True

    @torch.no_grad()
    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized forward pass through the NN. x shape: (batch, input_dim)."""
        h = x
        for w, b, act in zip(self.nn_weights, self.nn_biases,
                             self.activations):
            h = h @ w + b
            if act != "none":
                h = TORCH_ACTIVATIONS[act](h)
        return h.squeeze(-1)  # (batch,)

    def _ensure_initialized(self, input_dim: int):
        """Initialize if not yet done. Fast path when already initialized."""
        if self._initialized:
            return  # Fast return - no checks needed
        self._initialize(input_dim)

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        if len(parents) == 0:
            self._ensure_initialized(self.config.get("noise_input_dim", 1))
            x = torch.tensor([[noise]], dtype=torch.float32)
            return self._forward_batch(x).item()
        else:
            self._ensure_initialized(len(parents))
            x = torch.from_numpy(parents).float().unsqueeze(0)
            return self._forward_batch(x).item() + noise

    def forward_batch_torch(self,
                            parent_values: torch.Tensor,
                            noise: torch.Tensor,
                            is_root: bool = False) -> torch.Tensor:
        """
        Torch-native vectorized forward pass (no numpy conversion).

        Args:
            parent_values: (n_samples, n_parents) tensor, ignored if is_root
            noise: (n_samples,) tensor
            is_root: if True, pass noise through NN instead of parents

        Returns:
            values: (n_samples,) tensor
        """
        if is_root:
            self._ensure_initialized(self.config.get("noise_input_dim", 1))
            x = noise.unsqueeze(-1)  # (n_samples, 1)
            return self._forward_batch(x)
        else:
            self._ensure_initialized(parent_values.shape[1])
            return self._forward_batch(parent_values) + noise

    def copy(self) -> "NeuralNetEquation":
        return NeuralNetEquation(
            nn_weights=([w.clone() for w in self.nn_weights]
                        if self.nn_weights else None),
            nn_biases=[b.clone()
                       for b in self.nn_biases] if self.nn_biases else None,
            activations=self.activations.copy() if self.activations else None,
            layer_units=self.layer_units.copy() if self.layer_units else None,
            config=self.config,
            n_parents=self.n_parents,
        )


class GPEquation(StructuralEquation):
    """
    GP-based structural equation with a random kernel (RBF / Matérn-1/2 / Matérn-3/2).

    Inspired by CausalStructureNeuralProcess: uses GP priors to produce a
    much wider class of non-linear functions than fixed-architecture NNs.

    Key properties:
    - Lengthscale drawn from LogUniform(0.1, 5.0) at first call
    - Output scale drawn from Uniform(0.5, 2.0) at first call
    - A N(0,1) latent variable is concatenated to parent inputs (models
      unobserved confounders, following CSNP's latent-variable trick)
    - Noise scale drawn from Gamma(2.5, 0.4) per call (heteroscedastic)

    Uses forward_batch() for vectorized sampling; __call__ is a stub
    (per-sample GP is not well-defined without context).
    """

    def __init__(self):
        self.lengthscale: Optional[float] = None
        self.output_scale: Optional[float] = None
        self.kernel_type: Optional[str] = None

    def _init_hyperparams(self):
        self.lengthscale = np.exp(np.random.uniform(np.log(0.1), np.log(5.0)))
        self.output_scale = np.random.uniform(0.5, 2.0)
        self.kernel_type = np.random.choice(["rbf", "matern12", "matern32"])

    def _kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute (n, n) kernel matrix from (n, d) input."""
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, d)
        dist2 = np.sum(diff**2, axis=-1)  # (n, n)
        ell = self.lengthscale
        if self.kernel_type == "rbf":
            return np.exp(-0.5 * dist2 / ell**2)
        elif self.kernel_type == "matern12":
            dist = np.sqrt(dist2 + 1e-10)
            return np.exp(-dist / ell)
        else:  # matern32
            dist = np.sqrt(dist2 + 1e-10) * np.sqrt(3) / ell
            return (1.0 + dist) * np.exp(-dist)

    def forward_batch(self, parent_values: Optional[np.ndarray],
                      noise: np.ndarray) -> np.ndarray:
        """
        Sample GP output for a batch of parent values.

        Args:
            parent_values: (n_samples, n_parents) or None for root nodes
            noise: (n_samples,) pre-sampled noise (used for root nodes)

        Returns:
            (n_samples,) output values
        """
        if self.lengthscale is None:
            self._init_hyperparams()

        n_samples = noise.shape[0]

        if parent_values is None or parent_values.shape[1] == 0:
            # Root node: output is noise only (no parents to condition on)
            noise_scale = np.random.gamma(2.5, 0.4)
            return noise_scale * noise

        # Append a N(0,1) latent variable to parent inputs.
        # This models unobserved confounders (following CSNP latent trick).
        latent = np.random.standard_normal((n_samples, 1))
        X = np.hstack([parent_values, latent])  # (n_samples, n_parents + 1)

        # Normalize inputs for numerical stability
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = X / X_std

        # Compute kernel matrix and sample from GP prior: y ~ N(0, K)
        K = self._kernel(X_norm)
        K += 1e-3 * np.eye(n_samples)  # jitter for Cholesky stability

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-1 * np.eye(n_samples)
            L = np.linalg.cholesky(K)

        z = np.random.standard_normal(n_samples)
        y = self.output_scale * (L @ z)

        # Add Gamma-heteroscedastic noise (CSNP-style)
        noise_scale = np.random.gamma(2.5, 0.4)  # mean noise std ≈ 1.0
        return y + noise_scale * noise

    def __call__(self, parents: np.ndarray, noise: float) -> float:
        # Per-sample call: fall back to noise (GP requires full batch)
        return float(noise)

    def copy(self) -> "GPEquation":
        new = GPEquation()
        new.lengthscale = self.lengthscale
        new.output_scale = self.output_scale
        new.kernel_type = self.kernel_type
        return new


# =============================================================================
# DAG Generation
# =============================================================================


def generate_dag(n_nodes: int,
                 edge_probability: float = 0.3,
                 graph_type: str = "erdos-renyi") -> nx.DiGraph:
    """
    Generate a DAG.

    Args:
        n_nodes: Number of nodes
        edge_probability: Probability of edge between nodes (ignored for 'chain'/'full')
        graph_type: Type of random graph ('erdos-renyi', 'chain', 'full',
                    'barabasi-albert')

    Returns:
        dag: NetworkX DiGraph
    """
    if graph_type == "erdos-renyi":
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i):
                if np.random.rand() < edge_probability:
                    adj[i, j] = 1
    elif graph_type == "barabasi-albert":
        # Generate with m=1 (minimum), then subsample to match ER density.
        # BA with m=1 produces n-1 edges; subsampling brings it to target.
        target_edges = max(
            1, round(edge_probability * n_nodes * (n_nodes - 1) / 2))
        ug = nx.barabasi_albert_graph(n_nodes, 1)
        all_undirected = list(ug.edges())
        if target_edges < len(all_undirected):
            keep_idx = np.random.choice(len(all_undirected),
                                        size=target_edges,
                                        replace=False)
            selected = [all_undirected[k] for k in keep_idx]
        else:
            selected = all_undirected
        adj = np.zeros((n_nodes, n_nodes))
        for u, v in selected:
            i, j = max(u, v), min(u, v)  # i > j → adj[i,j]=1 means j→i
            adj[i, j] = 1
    elif graph_type == "bipartite":
        # CSNP-style bipartite: ~20% source nodes cause ~80% sink nodes.
        # Scale edge prob so expected edges matches ER: n*(n-1)/2 vs top*bottom potential edges.
        top = max(1, int(0.2 * n_nodes))
        bottom = n_nodes - top
        max_dag_edges = n_nodes * (n_nodes - 1) / 2
        bipartite_ep = min(
            edge_probability * max_dag_edges / max(top * bottom, 1), 1.0)
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(top, n_nodes):  # sink (effect) nodes
            for j in range(top):  # source (cause) nodes
                if np.random.rand() < bipartite_ep:
                    adj[i, j] = 1  # source j → sink i
    elif graph_type == "chain":
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(1, n_nodes):
            adj[i, i - 1] = 1
    elif graph_type == "full":
        adj = np.tril(np.ones((n_nodes, n_nodes)), k=-1)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Random permutation to break ordering bias
    perm = np.random.permutation(n_nodes)
    adj = adj[perm][:, perm]

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj[i, j] != 0:
                dag.add_edge(j, i)  # Edge from j to i

    return dag


# =============================================================================
# SCM (Structural Causal Model) Class
# =============================================================================


class StructuralCausalModel:
    """Structural Causal Model for data generation."""

    def __init__(
        self,
        dag: nx.DiGraph,
        noise_scale: float = 1.0,
        equation_type: str = "linear",
        nn_config: Optional[dict] = None,
        noise_type: str = "random",
    ):
        """
        Initialize the SCM.

        Args:
            dag: A directed acyclic graph
            noise_scale: Scale of the noise distribution
            equation_type: 'linear', 'nonlinear', or 'nn'
            nn_config: Neural network config dict (for equation_type='nn')
            noise_type: Noise distribution to use for all nodes — one of
                        'gaussian', 'poisson', 'salt_and_pepper',
                        'truncated_exponential', or 'random' (picks
                        uniformly at random from all four per SCM).
        """
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Graph must be a DAG")

        self.dag = dag
        self.noise_scale = noise_scale
        self.equation_type = equation_type
        self.nn_config = nn_config
        self.noise_type = (np.random.choice(NOISE_TYPES)
                           if noise_type == "random" else noise_type)
        self.equations: Dict[int, StructuralEquation] = {}
        self.topological_order = list(nx.topological_sort(dag))

        # For linear equations: 25% chance of "high-baseline" mode that mimics
        # availability-style metrics (near-constant at ~90-100 with tiny noise),
        # so any fault produces a z-score that saturates the ±10 clip boundary —
        # matching the real availability signature without adding a new eq type.
        # Internally reuses ConstantBaselineEquation math (Y = baseline + w·(X-baseline) + ε).
        self.high_baseline = 0.0  # 0 = normal mode
        if equation_type == "linear" and np.random.random() < 0.25:
            self.high_baseline = np.random.uniform(90.0, 100.0)
            # Override noise_scale with a near-zero value so normal-op variance ≈ 0
            self.noise_scale = np.random.uniform(1e-5, 5e-3)

        # Initialize equations
        for node in self.dag.nodes:
            if equation_type == "linear":
                if self.high_baseline > 0:
                    # High-baseline mode: deviation-based linear, near-zero noise
                    self.equations[node] = ConstantBaselineEquation(
                        baseline=self.high_baseline,
                        noise_scale=self.noise_scale)
                else:
                    self.equations[node] = LinearEquation()
            elif equation_type == "nonlinear":
                self.equations[node] = NonlinearEquation()
            elif equation_type == "constant_baseline":
                baseline = np.random.uniform(80.0, 120.0)
                noise_scale = np.random.uniform(0.01, 0.2)
                self.equations[node] = ConstantBaselineEquation(
                    baseline=baseline, noise_scale=noise_scale)
            elif equation_type == "nn":
                n_parents = len(list(self.dag.predecessors(node)))
                self.equations[node] = NeuralNetEquation(config=nn_config,
                                                         n_parents=n_parents)
            elif equation_type == "gp":
                self.equations[node] = GPEquation()
            else:
                raise ValueError(f"Unknown equation type: {equation_type}")

        # Pre-initialize NN equations to avoid repeated initialization checks during sampling
        if equation_type == "nn":
            self._preinitialize_nn_equations()

    def _preinitialize_nn_equations(self):
        """Pre-initialize all NN equations based on their parent count."""
        for node in self.dag.nodes:
            eq = self.equations[node]
            if isinstance(eq, NeuralNetEquation):
                parents = list(self.dag.predecessors(node))
                if len(parents) == 0:
                    # Root node: initialize with noise input dimension
                    eq._initialize(eq.config.get("noise_input_dim", 1))
                else:
                    # Non-root: initialize with number of parents
                    eq._initialize(len(parents))

    def sample(self,
               n_samples: int,
               temporal_autocorr: float = 0.0) -> np.ndarray:
        """
        Sample data from the SCM.

        Args:
            n_samples: Number of samples to generate
            temporal_autocorr: AR(1) autocorrelation coefficient for noise (0 = i.i.d.,
                               0.9+ = highly correlated, mimicking slow time-series).

        Returns:
            data: Array of shape (n_samples, n_nodes)
        """
        n_nodes = len(self.dag.nodes)

        # Sample noise for all nodes using the SCM's chosen noise type
        noise_np = _sample_noise(self.noise_type, self.noise_scale,
                                 (n_samples, n_nodes))

        # Apply AR(1) temporal autocorrelation to noise if requested.
        # Preserves variance: noise_t = rho * noise_{t-1} + sqrt(1-rho^2) * eps_t
        if temporal_autocorr > 0.0:
            ar_noise = np.zeros_like(noise_np)
            ar_noise[0] = noise_np[0]
            sqrt_term = np.sqrt(max(0.0, 1.0 - temporal_autocorr**2))
            for t in range(1, n_samples):
                ar_noise[t] = temporal_autocorr * ar_noise[
                    t - 1] + sqrt_term * noise_np[t]
            noise_np = ar_noise

        # Apply per-node noise scale overrides (used for root node interventions)
        noise_overrides = getattr(self, "noise_scale_overrides", {})
        for node, scale in noise_overrides.items():
            noise_np[:, node] = _sample_noise(self.noise_type, scale,
                                              n_samples)

        shift_overrides = getattr(self, "shift_overrides", {})
        do_overrides = getattr(self, "do_overrides",
                               {})  # hard do-interventions (constant value)
        use_torch = self.equation_type == "nn"

        if use_torch:
            # All-torch path: avoid numpy<->torch conversions per node
            data = torch.zeros(n_samples, n_nodes)
            noise = torch.from_numpy(noise_np).float()

            for node in self.topological_order:
                if node in do_overrides:
                    data[:, node] = do_overrides[
                        node]  # pin to constant (zero variance)
                    continue
                parents = list(self.dag.predecessors(node))
                eq = self.equations[node]

                if len(parents) == 0:
                    data[:, node] = eq.forward_batch_torch(None,
                                                           noise[:, node],
                                                           is_root=True)
                else:
                    parent_values = data[:, parents]
                    data[:, node] = eq.forward_batch_torch(parent_values,
                                                           noise[:, node],
                                                           is_root=False)
                if node in shift_overrides:
                    data[:, node] += shift_overrides[node]

            return data.numpy()
        else:
            # Numpy path for linear/nonlinear (vectorized where possible)
            data = np.zeros((n_samples, n_nodes))

            for node in self.topological_order:
                if node in do_overrides:
                    data[:, node] = do_overrides[
                        node]  # pin to constant (zero variance)
                    continue
                parents = list(self.dag.predecessors(node))
                eq = self.equations[node]

                if len(parents) == 0:
                    if isinstance(eq, ConstantBaselineEquation):
                        data[:,
                             node] = eq.baseline + noise_np[:,
                                                            node] * eq.noise_scale
                    else:
                        data[:, node] = noise_np[:, node]
                else:
                    if isinstance(eq, LinearEquation):
                        # Ensure weights are initialized
                        if eq.weights is None:
                            eq(data[0, parents], noise_np[0, node])
                        parent_values = data[:,
                                             parents]  # (n_samples, n_parents)

                        # CSNP: normalize inputs column-wise before applying weights
                        pmean = parent_values.mean(axis=0, keepdims=True)
                        pstd = parent_values.std(axis=0, keepdims=True) + 1e-8
                        parent_norm = (parent_values - pmean) / pstd

                        # CSNP: inject a per-node latent confounder (unobserved variable)
                        lat = np.random.standard_normal(n_samples)

                        # CSNP: Gamma(2.5, 2.5) heteroscedastic noise hyperprior
                        noise_hyper = np.random.gamma(shape=2.5, scale=2.5)
                        data[:, node] = (parent_norm @ eq.weights +
                                         eq.latent_weight * lat +
                                         noise_hyper * noise_np[:, node])
                    elif isinstance(eq, ConstantBaselineEquation):
                        # Vectorized constant baseline (no latent/norm — preserves availability pattern)
                        if eq.weights is None:
                            eq(data[0, parents], noise_np[0, node])
                        parent_values = data[:, parents]
                        parent_deviations = parent_values - eq.baseline
                        data[:,
                             node] = eq.baseline + parent_deviations @ eq.weights + noise_np[:,
                                                                                             node] * eq.noise_scale
                    elif isinstance(eq, GPEquation):
                        # Vectorized GP: samples entire time-series at once from kernel
                        parent_vals = data[:, parents] if len(
                            parents) > 0 else None
                        data[:,
                             node] = eq.forward_batch(parent_vals,
                                                      noise_np[:, node])
                    else:
                        # Per-sample loop for NonlinearEquation
                        for i in range(n_samples):
                            data[i, node] = eq(data[i, parents],
                                               noise_np[i, node])
                        # CSNP: inject latent confounder for nonlinear too
                        lat = np.random.standard_normal(n_samples)
                        data[:, node] += eq.latent_weight * lat
                if node in shift_overrides:
                    data[:, node] += shift_overrides[node]

            return data

    def copy(self) -> "StructuralCausalModel":
        """Create a deep copy of the SCM."""
        new_scm = StructuralCausalModel(
            self.dag.copy(),
            self.noise_scale,
            self.equation_type,
            nn_config=self.nn_config,
            noise_type=self.noise_type,  # preserve the sampled noise type
        )
        new_scm.equations = {
            node: eq.copy()
            for node, eq in self.equations.items()
        }
        new_scm.topological_order = self.topological_order.copy()
        new_scm.high_baseline = self.high_baseline
        if hasattr(self, "shift_overrides"):
            new_scm.shift_overrides = dict(self.shift_overrides)
        if hasattr(self, "noise_scale_overrides"):
            new_scm.noise_scale_overrides = dict(self.noise_scale_overrides)
        if hasattr(self, "do_overrides"):
            new_scm.do_overrides = dict(self.do_overrides)
        return new_scm


# =============================================================================
# Soft Interventions (Edge Weight Changes)
# =============================================================================


def apply_weight_intervention(
    scm: StructuralCausalModel,
    intervention_node: int,
    weight_change_factor: float = 3.0,
    flip_signs: bool = True,
) -> StructuralCausalModel:
    """
    Apply a strong soft intervention by changing edge weights.

    For linear/nonlinear equations: scale weights and optionally flip signs.
    For NN equations: change activation functions to create distribution shift.

    Args:
        scm: Original SCM
        intervention_node: Node to intervene on
        weight_change_factor: Multiply weights by this factor
        flip_signs: If True, flip the signs of some weights

    Returns:
        intervened_scm: New SCM with modified weights
    """
    intervened_scm = scm.copy()
    equation = intervened_scm.equations[intervention_node]

    if isinstance(equation, NeuralNetEquation):
        # NN equation: randomly pick ONE of two intervention types (never both)
        use_activation_change = np.random.random() < 0.5

        if use_activation_change and equation.activations is not None:
            # Change activation functions
            possible = equation.config.get("possible_activations",
                                           ["sigmoid", "tanh", "relu"])
            new_activations = []
            for act in equation.activations:
                if act == "none":
                    new_activations.append("none")
                else:
                    other_acts = [a for a in possible if a != act]
                    new_activations.append(np.random.choice(other_acts))
            equation.activations = new_activations
        elif equation.nn_weights is not None:
            # Scale NN weights (same approach as linear)
            for i in range(len(equation.nn_weights)):
                equation.nn_weights[
                    i] = equation.nn_weights[i] * weight_change_factor
                if flip_signs:
                    flip_mask = torch.rand(equation.nn_weights[i].shape) < 0.5
                    equation.nn_weights[i][flip_mask] *= -1
    elif hasattr(equation, "weights") and equation.weights is not None:
        # Linear/Nonlinear: Scale weights significantly
        equation.weights = equation.weights * weight_change_factor

        # Flip signs of some weights for stronger effect
        if flip_signs:
            n_weights = len(equation.weights)
            flip_mask = np.random.choice([True, False],
                                         size=n_weights,
                                         p=[0.5, 0.5])
            equation.weights[flip_mask] *= -1
    else:
        # Root node (no parents, not yet initialized): intervene by changing noise scale
        if isinstance(equation, NeuralNetEquation):
            # For NN root nodes, initialize first then apply one intervention type
            equation._initialize(equation.config.get("noise_input_dim", 1))
            use_activation_change = np.random.random() < 0.5

            if use_activation_change:
                possible = equation.config.get("possible_activations",
                                               ["sigmoid", "tanh", "relu"])
                equation.activations = [(np.random.choice(
                    [a for a in possible
                     if a != act]) if act != "none" else "none")
                                        for act in equation.activations]
            else:
                for i in range(len(equation.nn_weights)):
                    equation.nn_weights[i] = (equation.nn_weights[i] *
                                              weight_change_factor)
                    if flip_signs:
                        flip_mask = torch.rand(
                            equation.nn_weights[i].shape) < 0.5
                        equation.nn_weights[i][flip_mask] *= -1
        else:
            intervened_scm.noise_scale_overrides = getattr(
                intervened_scm, "noise_scale_overrides", {})
            intervened_scm.noise_scale_overrides[intervention_node] = (
                scm.noise_scale * weight_change_factor)

    return intervened_scm


def apply_shift_intervention(
    scm: StructuralCausalModel,
    intervention_node: int,
    weight_change_factor: float = 3.0,
) -> StructuralCausalModel:
    """
    Apply a soft shift intervention: add a constant offset to one node's output.

    The offset propagates naturally through all descendants because it is applied
    inside the topological sampling loop before children read their parents.

    For normal SCMs:
        shift = ±Uniform(0.5, 2.0) * noise_scale * weight_change_factor

    For availability SCMs (equation_type == 'availability'):
        shift = -(baseline * Uniform(0.03, 0.15))   [always a drop, 3–15%]
        This mimics a fault that reduces a near-100% metric, creating an
        astronomically large z-score that clips to ±10 — matching real data.

    Args:
        scm: Original SCM
        intervention_node: Node to intervene on
        weight_change_factor: Controls shift magnitude for non-availability SCMs

    Returns:
        intervened_scm: New SCM with a constant offset on the intervention node
    """
    intervened_scm = scm.copy()

    if scm.high_baseline > 0:
        # High-baseline (availability-like) mode: drop proportional to the baseline
        # so that the z-score saturates the ±10 clip boundary (3–15% of baseline)
        shift = -(scm.high_baseline * np.random.uniform(0.03, 0.15))
    else:
        magnitude = np.random.uniform(
            0.5, 2.0) * scm.noise_scale * weight_change_factor
        shift = magnitude * np.random.choice([-1, 1])

    if not hasattr(intervened_scm, "shift_overrides"):
        intervened_scm.shift_overrides = {}
    intervened_scm.shift_overrides[intervention_node] = shift
    return intervened_scm


def apply_do_intervention(
        scm: StructuralCausalModel,
        intervention_node: int,
        obs_data: np.ndarray,
        stuck_magnitude_range: tuple = (2.0, 4.0),
) -> StructuralCausalModel:
    """
    Apply a hard do-intervention: pin one node to a constant value (zero variance).

    Mimics the "stuck sensor" failure pattern in causRCA/availability data:
    - The intervention node is frozen at a constant value for the entire fault window.
    - Its variance collapses to zero; all other nodes remain stochastic.
    - Descendants are still influenced by the pinned value through the causal graph.

    Args:
        scm: Original SCM
        intervention_node: Node to pin to constant
        obs_data: Pre-fault observational data of shape (n_obs, n_nodes), used to
                  compute the node's normal mean/std so the stuck value is a
                  meaningful outlier.
        stuck_magnitude_range: Range (min, max) of |z-score| for the stuck value
                               relative to the normal distribution, e.g. (2, 4)
                               means the node is pinned 2-4 std devs from its mean.

    Returns:
        intervened_scm: New SCM with do_overrides set for the intervention node
    """
    obs_mean = obs_data[:, intervention_node].mean()
    obs_std = obs_data[:, intervention_node].std() + 1e-8
    direction = np.random.choice([-1.0, 1.0])
    magnitude = np.random.uniform(stuck_magnitude_range[0],
                                  stuck_magnitude_range[1])
    stuck_value = obs_mean + direction * magnitude * obs_std

    intervened_scm = scm.copy()
    intervened_scm.do_overrides = {intervention_node: float(stuck_value)}
    return intervened_scm


# =============================================================================
# Main Generation Function
# =============================================================================


def generate_observational_and_interventional_data(
    n_nodes: int = 5,
    n_obs_samples: int = 1000,
    n_int_samples: int = 1000,
    intervention_node: Optional[int] = None,
    equation_type: str = "linear",
    graph_type: str = "erdos-renyi",
    edge_probability: float = 0.4,
    noise_scale: float = 0.5,
    weight_change_factor: float = 3.0,
    flip_signs: bool = True,
    seed: Optional[int] = None,
    nn_config: Optional[dict] = None,
    noise_type: str = "random",
    do_intervention: bool = False,
    stuck_magnitude_range: tuple = (2.0, 4.0),
    temporal_autocorr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, nx.DiGraph, int]:
    """
    Generate both observational and interventional data with strong distribution shifts.

    Args:
        n_nodes: Number of nodes in the DAG
        n_obs_samples: Number of observational samples
        n_int_samples: Number of interventional samples
        intervention_node: Node to intervene on (random non-leaf if None)
        equation_type: 'linear', 'nonlinear', or 'nn'
        graph_type: Type of DAG
        edge_probability: Edge probability for DAG generation
        noise_scale: Scale of the noise distribution
        weight_change_factor: How much to scale weights (larger = stronger shift)
        flip_signs: Whether to flip signs of weights
        seed: Random seed
        nn_config: Neural network config dict (for equation_type='nn')
        noise_type: One of 'gaussian', 'poisson', 'salt_and_pepper',
                    'truncated_exponential', or 'random' (chosen uniformly per SCM).
        do_intervention: If True, use hard do-intervention (stuck sensor) instead of
                         weight-change. The intervention node is pinned to a constant
                         value, producing zero variance in the fault window.
        stuck_magnitude_range: (min, max) std-dev offset for the stuck value when
                                do_intervention=True.
        temporal_autocorr: AR(1) coefficient for temporally correlated noise
                           (0 = i.i.d., 0.9+ = slow time-series). Applied to both
                           obs and int windows.

    Returns:
        obs_data: Observational data (n_obs_samples, n_nodes)
        int_data: Interventional data (n_int_samples, n_nodes)
        dag: The DAG
        intervention_node: The node that was intervened on
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate DAG
    dag = generate_dag(n_nodes, edge_probability, graph_type)

    # Create SCM (noise_type is fixed at SCM creation; both obs and int use the same type)
    scm = StructuralCausalModel(dag,
                                noise_scale,
                                equation_type,
                                nn_config=nn_config,
                                noise_type=noise_type)

    # Sample observational data
    obs_data = scm.sample(n_obs_samples, temporal_autocorr=temporal_autocorr)

    # Select intervention node (prefer non-leaf nodes)
    if intervention_node is None:
        non_leaf_nodes = [
            node for node in dag.nodes if len(list(dag.successors(node))) > 0
        ]
        if not non_leaf_nodes:
            # If all nodes are leaves, just pick one with parents
            nodes_with_parents = [
                node for node in dag.nodes if dag.in_degree(node) > 0
            ]
            if not nodes_with_parents:
                raise ValueError("DAG has no suitable intervention nodes")
            intervention_node = random.choice(nodes_with_parents)
        else:
            intervention_node = random.choice(non_leaf_nodes)

    # Apply intervention: hard do-intervention (stuck sensor) or soft weight-change
    if do_intervention:
        intervened_scm = apply_do_intervention(scm, intervention_node,
                                               obs_data, stuck_magnitude_range)
    else:
        intervened_scm = apply_weight_intervention(scm, intervention_node,
                                                   weight_change_factor,
                                                   flip_signs)

    # Sample interventional data
    int_data = intervened_scm.sample(n_int_samples,
                                     temporal_autocorr=temporal_autocorr)

    return obs_data, int_data, dag, intervention_node


# =============================================================================
# Utility Functions
# =============================================================================


def compare_distributions(obs_data: np.ndarray, int_data: np.ndarray,
                          intervention_node: int, dag: nx.DiGraph):
    """
    Compare distributions and run KS tests.

    Args:
        obs_data: Observational data
        int_data: Interventional data
        intervention_node: The intervened node
        dag: The DAG to identify descendants
    """
    n_nodes = obs_data.shape[1]

    # Find all descendants of the intervention node
    descendants = nx.descendants(dag, intervention_node)

    print("\nDistribution Comparison:")
    print("=" * 70)
    print(
        f"{'Node':<6} {'Obs Mean':<12} {'Int Mean':<12} {'KS Stat':<12} {'p-value':<12}"
    )
    print("-" * 70)

    for node in range(n_nodes):
        obs_node = obs_data[:, node]
        int_node = int_data[:, node]

        obs_mean = obs_node.mean()
        int_mean = int_node.mean()

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(obs_node, int_node)

        if node == intervention_node:
            marker = " <-- INTERVENED"
        elif node in descendants:
            marker = " <-- DESCENDANT"
        else:
            marker = ""

        print(
            f"{node:<6} {obs_mean:<12.3f} {int_mean:<12.3f} {ks_stat:<12.4f} {p_value:<12.6f}{marker}"
        )

    print("=" * 70)
    print("Note: p-value < 0.05 indicates significant distribution shift")


def visualize_dag(
    dag: nx.DiGraph,
    intervention_node: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Visualize the DAG with intervention node in red and descendants in orange."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(dag, seed=42)

    # Default color for all nodes
    node_colors = ["lightblue"] * len(dag.nodes)
    node_list = list(dag.nodes)

    if intervention_node is not None:
        # Find all descendants of the intervention node
        descendants = nx.descendants(dag, intervention_node)

        # Color descendants orange
        for node in descendants:
            if node in node_list:
                idx = node_list.index(node)
                node_colors[idx] = "orange"

        # Color intervention node red (do this last so it overrides)
        if intervention_node in node_list:
            idx = node_list.index(intervention_node)
            node_colors[idx] = "red"

    nx.draw(
        dag,
        pos,
        node_color=node_colors,
        node_size=800,
        with_labels=True,
        font_size=14,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        width=2,
    )

    title = "DAG Structure"
    if intervention_node is not None:
        title += (
            f" (Red = Intervention on node {intervention_node}, Orange = Descendants)"
        )
    plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distribution_comparison(
    obs_data: np.ndarray,
    int_data: np.ndarray,
    intervention_node: int,
    dag: nx.DiGraph,
    save_path: Optional[str] = None,
):
    """Plot histograms comparing distributions for intervention node and descendants only."""
    import matplotlib.pyplot as plt

    # Find all descendants of the intervention node
    descendants = nx.descendants(dag, intervention_node)

    nodes_to_plot = dag.nodes
    n_nodes = len(nodes_to_plot)

    if n_nodes == 0:
        print("No nodes to plot (intervention node has no descendants)")
        return

    n_cols = min(n_nodes, 3)
    n_rows = (n_nodes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_nodes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, node in enumerate(nodes_to_plot):
        ax = axes[plot_idx]
        ax.hist(obs_data[:, node],
                bins=40,
                alpha=0.6,
                label="Observational",
                density=True)
        ax.hist(int_data[:, node],
                bins=40,
                alpha=0.6,
                label="Interventional",
                density=True)

        title = f"Node {node}"
        if node == intervention_node:
            title += " (INTERVENED)"
            ax.set_facecolor("#ffcccc")  # Light red background
        elif node in descendants:
            title += " (DESCENDANT)"
            ax.set_facecolor("#ffe6cc")  # Light orange background
        else:
            title += " (Unaffected)"
            ax.set_facecolor("#f0f0f0")  # Light gray background

        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Hide extra subplots
    for i in range(n_nodes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
