import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset
import random
import networkx as nx
from prim.data.dag_generator_simplified import (
    generate_dag,
    StructuralCausalModel,
    apply_weight_intervention,
    apply_shift_intervention,
    apply_do_intervention,
    load_nn_config,
)


class ContextQueryDataGenerator:
    """
    Generate episodes with context + query from the same DAG

    Each episode:
    - Sample ONE DAG
    - Generate k_context samples from this DAG (unlabeled during training)
    - Generate 1 query sample from this DAG (labeled)

    Model learns: P(query_label | query_data, context_data)
    """

    def __init__(
            self,
            n_nodes_range=(5, 10),
            max_nodes=10,
            obs_range=(50, 300),
            int_range=(5, 50),
            k_context_range=(0, 10),  # Range: 0=zero-shot, 1+=few-shot
            n_queries=16,  # Number of query samples per episode
            equation_type="mixed",  # 'linear', 'nn', or 'mixed' (50/50)
            nn_config_path=None,
            expected_degree_range=(
                1.8,
                2.5,
            ),  # Average degree per node (controls graph density)
    ):
        self.n_nodes_range = n_nodes_range
        self.max_nodes = max_nodes
        self.obs_range = obs_range
        self.int_range = int_range
        self.k_context_range = k_context_range
        self.n_queries = n_queries
        self.equation_type = equation_type
        self.expected_degree_range = expected_degree_range
        self.nn_config = None
        if equation_type in ("nn", "mixed"):
            self.nn_config = load_nn_config(nn_config_path)

    def generate_episode(self):
        """
        Generate one episode from a single DAG

        Returns:
            context_obs: (k_context, n_obs, max_nodes) - context observational data
            context_int: (k_context, n_int, max_nodes) - context interventional data
            query_obs: (n_queries, n_obs, max_nodes) - query observational data
            query_int: (n_queries, n_int, max_nodes) - query interventional data
            query_labels: (n_queries,) - intervention nodes for all queries
        """

        # Sample DAG parameters ONCE for the entire episode
        k = np.random.randint(self.n_nodes_range[0], self.n_nodes_range[1] + 1)
        dag_family = random.choice(
            ["erdos-renyi", "barabasi-albert", "bipartite"])
        # Use expected degree to compute edge_prob: keeps density consistent across graph sizes
        # edge_prob = expected_degree / (n_nodes - 1)
        expected_degree = np.random.uniform(self.expected_degree_range[0],
                                            self.expected_degree_range[1])
        edge_prob = min(expected_degree / (k - 1), 0.99) if k > 1 else 0.5
        # Log-uniform noise scale: equal prior mass per order of magnitude (0.05 to 2.0)
        noise_scale = np.exp(np.random.uniform(np.log(0.05), np.log(2.0)))
        weight_change_factor = np.random.uniform(3.0, 5.0)

        # Sample sequence lengths once per episode
        n_obs = np.random.randint(self.obs_range[0], self.obs_range[1])
        n_int = np.random.randint(self.int_range[0], self.int_range[1])

        # Sample number of context samples for THIS episode
        k_context = np.random.randint(self.k_context_range[0],
                                      self.k_context_range[1] + 1)

        # Pick equation type for this episode
        if self.equation_type == "mixed":
            episode_eq_type = np.random.choice(
                ["linear", "nn", "constant_baseline", "gp"])
        else:
            episode_eq_type = self.equation_type

        # Generate ONE DAG and SCM for the entire episode
        for attempt in range(10):
            dag = generate_dag(k, edge_prob, dag_family)
            scm = StructuralCausalModel(
                dag,
                noise_scale,
                episode_eq_type,
                nn_config=self.nn_config if episode_eq_type == "nn" else None,
            )
            # Pre-initialize weights by sampling once
            _ = scm.sample(1)

            # Find valid intervention nodes
            non_leaf_nodes = [
                node for node in dag.nodes
                if len(list(dag.successors(node))) > 0
            ]
            if not non_leaf_nodes:
                nodes_with_parents = [
                    node for node in dag.nodes if dag.in_degree(node) > 0
                ]
                if not nodes_with_parents:
                    continue  # Retry with a new DAG
                valid_nodes = nodes_with_parents
            else:
                valid_nodes = non_leaf_nodes
            break
        else:
            raise ValueError(
                "Could not generate a suitable DAG after 10 attempts")

        context_obs = []
        context_int = []
        context_target_masks = []

        # Generate k_context samples from the SAME DAG
        for _ in range(k_context):
            obs, int_data, label, target_mask = self._generate_sample_from_scm(
                scm, dag, valid_nodes, k, n_obs, n_int, weight_change_factor)
            context_obs.append(obs)
            context_int.append(int_data)
            context_target_masks.append(target_mask)

        # Generate n_queries query samples from the SAME DAG
        query_obs = []
        query_int = []
        query_labels = []
        query_target_masks = []
        for _ in range(self.n_queries):
            obs, int_data, label, target_mask = self._generate_sample_from_scm(
                scm, dag, valid_nodes, k, n_obs, n_int, weight_change_factor)
            query_obs.append(obs)
            query_int.append(int_data)
            query_labels.append(label)
            query_target_masks.append(target_mask)

        # Stack query samples
        query_obs_stacked = np.stack(
            query_obs)  # (n_queries, n_obs, max_nodes)
        query_int_stacked = np.stack(
            query_int)  # (n_queries, n_int, max_nodes)
        query_labels = np.array(query_labels, dtype=np.int64)  # (n_queries,)
        query_target_masks_stacked = np.stack(
            query_target_masks)  # (n_queries, max_nodes)

        # Handle k_context=0 case (empty context)
        if k_context == 0:
            context_obs_stacked = np.empty((0, n_obs, self.max_nodes),
                                           dtype=np.float32)
            context_int_stacked = np.empty((0, n_int, self.max_nodes),
                                           dtype=np.float32)
            context_target_masks_stacked = np.empty((0, self.max_nodes),
                                                    dtype=np.float32)
        else:
            context_obs_stacked = np.stack(context_obs)
            context_int_stacked = np.stack(context_int)
            context_target_masks_stacked = np.stack(context_target_masks)

        return (
            context_obs_stacked,  # (k_context, n_obs, max_nodes)
            context_int_stacked,  # (k_context, n_int, max_nodes)
            query_obs_stacked,  # (n_queries, n_obs, max_nodes)
            query_int_stacked,  # (n_queries, n_int, max_nodes)
            query_labels,  # (n_queries,)
            query_target_masks_stacked,  # (n_queries, max_nodes)
            context_target_masks_stacked,  # (k_context, max_nodes)
            k,  # actual number of nodes in this DAG
        )

    def _generate_sample_from_scm(self, scm, dag, valid_nodes, k, n_obs, n_int,
                                  weight_change_factor):
        """Generate one (obs, int, label, target_mask) sample from an existing DAG/SCM"""

        # Sample fresh observational data from the same SCM
        obs_data = scm.sample(n_obs)

        # Pick a random intervention node and intervention type
        intervention_node = random.choice(valid_nodes)
        roll = np.random.random()
        if roll < 0.8:
            intervened_scm = apply_weight_intervention(scm,
                                                       intervention_node,
                                                       weight_change_factor,
                                                       flip_signs=True)
        elif roll < 0.95:
            intervened_scm = apply_shift_intervention(scm, intervention_node,
                                                      weight_change_factor)
        else:
            intervened_scm = apply_do_intervention(scm, intervention_node,
                                                   obs_data)
        int_data = intervened_scm.sample(n_int)

        # Pick a target (symptom) node: the intervention node itself or a descendant
        descendants = list(nx.descendants(dag, intervention_node))
        target_candidates = [intervention_node] + descendants
        target_node = random.choice(target_candidates)

        # Build target mask: (max_nodes,) with 1 at the target node position
        K = self.max_nodes
        target_mask = np.zeros(K, dtype=np.float32)
        target_mask[target_node] = 1.0

        # Normalize then clip to ±10 so near-zero-std data
        # mode) never produces exploding z-scores that corrupt model activations.
        # This also matches the implicit clip applied by real-data analysis pipelines.
        _CLIP = 10.0
        obs_mean = obs_data.mean(axis=0, keepdims=True)
        obs_std = obs_data.std(axis=0, keepdims=True) + 1e-8
        obs_normalized = np.clip((obs_data - obs_mean) / obs_std, -_CLIP,
                                 _CLIP)
        int_normalized = np.clip((int_data - obs_mean) / obs_std, -_CLIP,
                                 _CLIP)

        # # Simulate "inactive subsystem" nodes (~1% of episodes).
        # # These nodes are structurally present but produce constant readings
        # # (e.g., unaffected subsystems in full-graph evaluation).
        # # Constraint: silent nodes CANNOT be the intervention node or its descendants,
        # # because those nodes are causally affected during the fault period.
        # if np.random.random() < 0.01:
        #     protected = set([intervention_node] + descendants)
        #     silent_candidates = [n for n in range(k) if n not in protected]
        #     if len(silent_candidates) >= 1:
        #         max_silent = max(1, len(silent_candidates) // 2)
        #         n_silent = np.random.randint(1, max_silent + 1)
        #         silent_nodes = np.random.choice(
        #             silent_candidates, n_silent, replace=False
        #         )
        #         obs_normalized[:, silent_nodes] = (
        #             0.0  # constant at mean (zero after z-score)
        #         )
        #         int_normalized[:, silent_nodes] = 0.0  # unchanged during fault period

        scale_factor = K / k

        obs_padded = np.zeros((n_obs, K), dtype=np.float32)
        int_padded = np.zeros((n_int, K), dtype=np.float32)

        obs_padded[:, :k] = obs_normalized * scale_factor
        int_padded[:, :k] = int_normalized * scale_factor

        return obs_padded, int_padded, intervention_node, target_mask


class ContextQueryDataset(IterableDataset):
    """
    Dataset that generates episodes with context + query

    Each episode = k_context + 1 query samples from the SAME DAG
    """

    def __init__(
            self,
            episodes_per_epoch=1000,
            k_context_range=(0, 10),
            n_nodes_range=(5, 10),
            max_nodes=10,
            obs_range=(100, 300),
            int_range=(50, 200),
            n_queries=32,
            seed=None,
            equation_type="mixed",
            nn_config_path=None,
            expected_degree_range=(1.8, 2.5),
            prior="scm",
            single_rc=False,
    ):
        self.episodes_per_epoch = episodes_per_epoch
        self.k_context_range = k_context_range
        # `prior` selects the data-generating process:
        #   "scm"    -> ContextQueryDataGenerator (dag_generator_simplified)
        #   "tabicl" -> TabICLRCADataGenerator (prim.data.tabicl_rca_prior):
        #               TabICL-style mixed MLP/tree SCMs + discrete columns
        gen_kwargs = dict(
            equation_type=equation_type,
            nn_config_path=nn_config_path,
            expected_degree_range=expected_degree_range,
        )
        if prior == "tabicl":
            from prim.data.tabicl_rca_prior import (
                TabICLRCADataGenerator,
                TabICLPriorConfig,
            )

            gen_cls = TabICLRCADataGenerator
            # `single_rc` forces every episode to have exactly one root cause
            # (baseline for the multi-RC-training A/B); default keeps the
            # config's 70/20/10 single/dual/triple mix.
            if single_rc:
                gen_kwargs["config"] = TabICLPriorConfig(
                    n_nodes_range=n_nodes_range,
                    max_nodes=max_nodes,
                    expected_degree_range=expected_degree_range,
                    n_faults_probs=(1.0, 0.0, 0.0),
                )
        elif prior == "scm":
            gen_cls = ContextQueryDataGenerator
        else:
            raise ValueError(
                f"unknown prior: {prior!r} (use 'scm' or 'tabicl')")
        self.data_generator = gen_cls(
            n_nodes_range,
            max_nodes,
            obs_range,
            int_range,
            k_context_range,
            n_queries,
            **gen_kwargs,
        )
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Seed per worker — stride by num_workers so consecutive epochs never share seeds.
        # Formula: epoch_base * num_workers + worker_id guarantees unique seeds for any
        # num_workers value (tested with 8 workers and multiple GPU ranks).
        if self.seed is not None:
            seed = self.seed * num_workers + worker_id
            np.random.seed(seed % (2**31))
            random.seed(seed)

        episodes_per_worker = self.episodes_per_epoch // num_workers

        for _ in range(episodes_per_worker):
            (
                context_obs,
                context_int,
                query_obs,
                query_int,
                query_labels,
                query_target_masks,
                context_target_masks,
                n_nodes,
            ) = self.data_generator.generate_episode()

            yield (
                torch.from_numpy(context_obs),  # (k_context, n_obs, max_nodes)
                torch.from_numpy(context_int),  # (k_context, n_int, max_nodes)
                torch.from_numpy(query_obs),  # (n_queries, n_obs, max_nodes)
                torch.from_numpy(query_int),  # (n_queries, n_int, max_nodes)
                torch.from_numpy(query_labels),  # (n_queries,)
                torch.from_numpy(query_target_masks),  # (n_queries, max_nodes)
                torch.from_numpy(
                    context_target_masks),  # (k_context, max_nodes)
                torch.tensor(n_nodes,
                             dtype=torch.long),  # scalar: actual node count
            )

    def __len__(self):
        return self.episodes_per_epoch


def rca_loss_and_acc(query_logits,
                     query_labels,
                     query_target_masks,
                     alarm_penalty=0.0):
    """Ranking loss + top-1 accuracy for query labels that are EITHER a 1-D
    LongTensor of class indices (single root cause) OR a 2-D multi-hot
    FloatTensor (one or more independent root causes).

    Multi-hot loss = mean over the true root causes of the softmax
    cross-entropy toward each (cf. prim/finetuning/finetune_causrca.py
    train_step).  For a single root cause this is exactly F.cross_entropy, so
    the `scm` prior / PetShop behaviour is unchanged.

    Accuracy = "the top-1 predicted node is one of the true root causes".
    """
    K = query_logits.shape[-1]
    if query_labels.dim() == 1:
        multi_hot = F.one_hot(query_labels.long(), K).to(query_logits.dtype)
    else:
        multi_hot = query_labels.to(query_logits.dtype)

    logp = F.log_softmax(query_logits, dim=-1)
    # -inf at padded positions * 0 (not a positive) would be nan -> mask first
    contrib = (-logp).masked_fill(multi_hot == 0, 0.0)
    per_q = contrib.sum(-1) / multi_hot.sum(-1).clamp(min=1.0)
    loss = per_q.mean()

    # Alarm-avoidance penalty (cf. finetune_causrca): push the observed-symptom
    # logit down so the model can't shortcut to "predict the biggest z-score".
    # Skipped for queries where the symptom is itself a root cause.
    if alarm_penalty > 0.0:
        sym_idx = query_target_masks.argmax(dim=1)
        sym_is_rc = multi_hot.gather(1, sym_idx.unsqueeze(1)).squeeze(1) > 0
        keep = ~sym_is_rc
        if keep.any():
            sym_logit = query_logits.gather(1, sym_idx.unsqueeze(1)).squeeze(1)
            loss = loss + alarm_penalty * F.softplus(sym_logit[keep]).mean()

    with torch.no_grad():
        pred = query_logits.argmax(dim=1)
        acc = ((multi_hot.gather(1, pred.unsqueeze(1)).squeeze(1)
                > 0).float().mean().item())
    return loss, acc


def train_step_context(
    model,
    context_obs,
    context_int,
    query_obs,
    query_int,
    query_labels,
    query_target_masks,
    context_target_masks,
    n_nodes,
    optimizer,
    alarm_penalty=0.0,
):
    """
    Training step with context + multiple queries WITH MASKING

    Args:
        context_obs: (k_context, n_obs, max_nodes)
        context_int: (k_context, n_int, max_nodes)
        query_obs: (n_queries, n_obs, max_nodes)
        query_int: (n_queries, n_int, max_nodes)
        query_labels: (n_queries,)
        query_target_masks: (n_queries, max_nodes)
        context_target_masks: (k_context, max_nodes)
        n_nodes: scalar - actual number of nodes in this DAG
        optimizer: The optimizer
        alarm_penalty: weight of the alarm-avoidance loss term (see multigpu)
    """
    model.train()
    optimizer.zero_grad()

    n_queries = query_obs.shape[0]
    max_nodes = query_obs.shape[-1]

    # Concatenate context and queries
    all_obs = torch.cat([context_obs, query_obs],
                        dim=0)  # (k_context+n_queries, n_obs, max_nodes)
    all_int = torch.cat([context_int, query_int],
                        dim=0)  # (k_context+n_queries, n_int, max_nodes)
    all_target_masks = torch.cat([context_target_masks, query_target_masks],
                                 dim=0)  # (k_context+n_queries, max_nodes)

    # Forward pass
    logits = model(
        all_obs, all_int,
        target_mask=all_target_masks)  # (k_context+n_queries, max_nodes)

    # Compute loss on all queries (last n_queries samples)
    query_logits = logits[-n_queries:]  # (n_queries, max_nodes)

    # MASK PADDED POSITIONS: Set logits to -inf for positions >= n_nodes
    # This prevents the model from predicting invalid node positions!
    if n_nodes < max_nodes:
        mask = torch.zeros_like(query_logits)
        mask[:, n_nodes:] = float("-inf")
        query_logits = query_logits + mask

    loss, correct = rca_loss_and_acc(query_logits, query_labels,
                                     query_target_masks, alarm_penalty)

    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), correct


def train_context_meta_learning(
    model,
    device,
    num_epochs=5,
    episodes_per_epoch=1000,
    k_context_range=(0, 10),
    n_queries=16,
    lr=5e-4,
    equation_type="mixed",
    min_nodes=5,
    max_nodes=50,
    prior="scm",
    alarm_penalty=0.0,
    single_rc=False,
):
    """
    Train with context + query episodes

    Each episode:
    - k_context samples from same DAG (context) - k_context varies per episode!
      - k=0: Zero-shot (no context)
      - k>=1: Few-shot (with context)
    - n_queries query samples from same DAG
    - Predict all query labels using context
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("=" * 70)
    print("CONTEXT-BASED META-LEARNING (Zero-shot + Few-shot)")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Episodes per epoch: {episodes_per_epoch}")
    print(
        f"Context samples per episode: {k_context_range[0]}-{k_context_range[1]} (varies)"
    )
    print("  - k=0: Zero-shot learning")
    print("  - k≥1: Few-shot in-context learning")
    print(f"Query samples per episode: {n_queries}")
    print(f"Total labels per epoch: {episodes_per_epoch * n_queries:,}")
    print(f"Equation type: {equation_type}")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(
            f"\nEpoch {epoch+1}/{num_epochs}  [graph size: {min_nodes}–{max_nodes} nodes, uniform]"
        )

        # Create dataset
        dataset = ContextQueryDataset(
            episodes_per_epoch=episodes_per_epoch,
            k_context_range=k_context_range,
            n_queries=n_queries,
            seed=42 + epoch,
            equation_type=equation_type,
            n_nodes_range=(min_nodes, max_nodes),
            max_nodes=max_nodes,
            prior=prior,
            single_rc=single_rc,
        )

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=None,
                                                   num_workers=4,
                                                   pin_memory=True)

        epoch_loss = 0
        epoch_acc = 0

        from tqdm import tqdm

        for episode_idx, (
                context_obs,
                context_int,
                query_obs,
                query_int,
                query_labels,
                query_target_masks,
                context_target_masks,
                n_nodes,
        ) in enumerate(tqdm(train_loader)):

            context_obs = context_obs.to(device)
            context_int = context_int.to(device)
            query_obs = query_obs.to(device)
            query_int = query_int.to(device)
            query_labels = query_labels.to(device)
            query_target_masks = query_target_masks.to(device)
            context_target_masks = context_target_masks.to(device)
            n_nodes = n_nodes.item()  # Convert to Python int

            loss, acc = train_step_context(
                model,
                context_obs,
                context_int,
                query_obs,
                query_int,
                query_labels,
                query_target_masks,
                context_target_masks,
                n_nodes,
                optimizer,
                alarm_penalty=alarm_penalty,
            )

            epoch_loss += loss
            epoch_acc += acc

        avg_loss = epoch_loss / episodes_per_epoch
        avg_acc = epoch_acc / episodes_per_epoch

        print(
            f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"context_meta_epoch_{epoch+1}.pt")

    print("\n✓ Context-based meta-learning complete!")
    return model


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Context meta-learning (single GPU)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--episodes_per_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_nodes", type=int, default=15)
    parser.add_argument("--max_nodes", type=int, default=100)
    parser.add_argument("--embedding_size", type=int, default=160)
    parser.add_argument("--equation_type", type=str, default="mixed")
    parser.add_argument("--n_queries", type=int, default=4)
    parser.add_argument("--k_context_min", type=int, default=0)
    parser.add_argument("--k_context_max", type=int, default=0)
    parser.add_argument(
        "--prior",
        type=str,
        default="scm",
        choices=["scm", "tabicl"],
        help="Data-generating prior: 'scm' (default) or 'tabicl' "
        "(TabICL-style mixed MLP/tree SCMs + discrete columns)",
    )
    parser.add_argument(
        "--alarm_penalty",
        type=float,
        default=0.0,
        help="Weight of the alarm-avoidance loss term (cf. finetune_causrca); "
        "try 0.1-0.25 with --prior tabicl.",
    )
    parser.add_argument(
        "--single_rc",
        action="store_true",
        help="Force exactly one root cause per episode (--prior tabicl only; "
        "single-RC baseline for the multi-RC-training A/B).",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from prim.model.prim_neural_process import PRIMNeuralProcess

    model = PRIMNeuralProcess(
        embedding_size=args.embedding_size,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=args.max_nodes,
        dropout=0.1,
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_context_meta_learning(
        model,
        device,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        k_context_range=(args.k_context_min, args.k_context_max),
        n_queries=args.n_queries,
        lr=args.lr,
        equation_type=args.equation_type,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        prior=args.prior,
        alarm_penalty=args.alarm_penalty,
        single_rc=args.single_rc,
    )

    torch.save(model.state_dict(), "context_meta_final.pt")
    print("\n✓ Model saved: context_meta_final.pt")


if __name__ == "__main__":
    main()
