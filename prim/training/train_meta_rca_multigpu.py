"""
Multi-GPU Context-based Meta-Learning for Intervention Detection

Each episode: k_context + 1 query samples from the SAME DAG
Model learns to use context to predict query (in-context learning!)
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from prim.training.train_meta_rca import ContextQueryDataset, rca_loss_and_acc


def _single_pass_predict(
    model,
    device,
    context_obs,
    context_int,
    query_obs,
    query_int,
    query_target_mask,
    context_target_masks,
    n_nodes,
    max_nodes,
):
    """Single eval-mode forward pass returning ranked node indices."""
    model.eval()
    with torch.no_grad():
        # Build tensors: (k+1, T, max_nodes)
        if context_obs.shape[0] > 0:
            all_obs = torch.cat([context_obs, query_obs], dim=0)
            all_int = torch.cat([context_int, query_int], dim=0)
            all_masks = torch.cat([context_target_masks, query_target_mask],
                                  dim=0)
        else:
            all_obs = query_obs
            all_int = query_int
            all_masks = query_target_mask

        logits = model(all_obs,
                       all_int,
                       target_mask=all_masks,
                       n_nodes=n_nodes)
        q_logits = logits[-query_obs.shape[0]:]  # last n_query rows

        if n_nodes < max_nodes:
            pad_mask = torch.zeros_like(q_logits)
            pad_mask[:, n_nodes:] = float("-inf")
            q_logits = q_logits + pad_mask

        probs = torch.softmax(q_logits, dim=-1).cpu().numpy()
    model.train()
    return probs


def run_real_data_validation(model, device, max_nodes, epoch):
    """
    Run validation on PetShop train data after each epoch.
    Uses a single eval-mode forward pass (no MC dropout).
    Only call this on rank 0.
    CausRCA is excluded — those runs are reserved as test data.

    Returns a dict of metrics (also logs to wandb if a run is active).
    """

    unwrapped = model.module if hasattr(model, "module") else model
    metrics = {}

    # ── PetShop (train split) ─────────────────────────────────────────────
    try:
        import prim.evaluation.test_petshop as petshop_mod

        petshop_mod.MAX_NODES = max_nodes
        print("\n  [Val] PetShop train-split evaluation...")

        top1_hits, top3_hits, total = 0, 0, 0
        for scenario in petshop_mod.SCENARIOS:
            scenario_path = petshop_mod.os.path.join(petshop_mod.DATASET_DIR,
                                                     scenario)
            graph, normal_metrics, issues = petshop_mod.load_scenario(
                scenario_path)
            for abnormal_metrics, target in issues["train"]:
                target_node = target["target"]["node"]
                target_metric = target["target"]["metric"]
                root_cause_node = target["root_cause"]["node"]
                metric, statistic = petshop_mod.METRIC_CONFIG.get(
                    target_metric, ("latency", "p95"))
                node_names = petshop_mod.get_node_names(
                    normal_metrics, metric, statistic)
                n_nodes = len(node_names)

                query_obs, query_int = petshop_mod.petshop_to_model_input(
                    normal_metrics, abnormal_metrics, node_names, metric,
                    statistic)
                query_target_mask = np.zeros(max_nodes, dtype=np.float32)
                if target_node in node_names:
                    query_target_mask[node_names.index(target_node)] = 1.0

                q_obs_t = torch.from_numpy(query_obs).unsqueeze(0).to(device)
                q_int_t = torch.from_numpy(query_int).unsqueeze(0).to(device)
                q_mask_t = torch.from_numpy(query_target_mask).unsqueeze(0).to(
                    device)
                ctx_obs_t = torch.empty(0,
                                        query_obs.shape[0],
                                        max_nodes,
                                        device=device)
                ctx_int_t = torch.empty(0,
                                        query_int.shape[0],
                                        max_nodes,
                                        device=device)
                ctx_mask_t = torch.empty(0, max_nodes, device=device)

                probs = _single_pass_predict(
                    unwrapped,
                    device,
                    ctx_obs_t,
                    ctx_int_t,
                    q_obs_t,
                    q_int_t,
                    q_mask_t,
                    ctx_mask_t,
                    n_nodes,
                    max_nodes,
                )
                ranked = np.argsort(-probs[0])
                ranked_nodes = [node_names[i] for i in ranked if i < n_nodes]
                top1_hits += int(root_cause_node in ranked_nodes[:1])
                top3_hits += int(root_cause_node in ranked_nodes[:3])
                total += 1

        if total > 0:
            metrics["val/petshop_train_top1"] = top1_hits / total
            metrics["val/petshop_train_top3"] = top3_hits / total
            print(
                f"  [Val] PetShop train  top1={metrics['val/petshop_train_top1']:.3f}"
                f"  top3={metrics['val/petshop_train_top3']:.3f}  n={total}")
    except Exception as e:
        print(f"  [Val] PetShop validation failed: {e}")

    # ── Wandb logging ─────────────────────────────────────────────────────
    if metrics:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"epoch": epoch + 1, **metrics})
        except ImportError:
            pass

    return metrics


def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device


def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()


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
    """

    n_queries = query_obs.shape[0]
    max_nodes = query_obs.shape[-1]

    # Concatenate context and queries
    all_obs = torch.cat([context_obs, query_obs],
                        dim=0)  # (k_context+n_queries, n_obs, max_nodes)
    all_int = torch.cat([context_int, query_int],
                        dim=0)  # (k_context+n_queries, n_int, max_nodes)
    all_target_masks = torch.cat([context_target_masks, query_target_masks],
                                 dim=0)  # (k_context+n_queries, max_nodes)

    # Forward pass - model processes all samples together
    logits = model(all_obs,
                   all_int,
                   target_mask=all_target_masks,
                   n_nodes=n_nodes)  # (k_context+n_queries, max_nodes)

    # Compute loss on all queries (last n_queries samples)
    query_logits = logits[-n_queries:]  # (n_queries, max_nodes)

    # MASK PADDED POSITIONS: Set logits to -inf for positions >= n_nodes
    # This prevents the model from predicting invalid node positions!
    if n_nodes < max_nodes:
        mask = torch.zeros_like(query_logits)
        mask[:, n_nodes:] = float("-inf")
        query_logits = query_logits + mask

    # Ranking loss + top-1 accuracy; handles single-label (1-D) and multi-hot
    # (2-D, one or more independent root causes) query labels, plus the
    # alarm-avoidance penalty (see train_meta_rca.rca_loss_and_acc).
    loss, correct = rca_loss_and_acc(query_logits, query_labels,
                                     query_target_masks, alarm_penalty)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), correct


def train_context_meta_ddp(
    model,
    device,
    rank,
    world_size,
    num_epochs=5,
    episodes_per_epoch=1000,
    k_context_range=(0, 0),
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
    Context-based meta-learning with multi-GPU support.
    Each episode: k_context samples + n_queries from the SAME DAG.
    k_context varies per episode: 0=zero-shot, 1+=few-shot
    """

    # Conservative LR scaling for transformer
    scaled_lr = lr * (world_size**0.5)
    optimizer = optim.AdamW(model.parameters(),
                            lr=scaled_lr,
                            weight_decay=0.01)

    if rank == 0:
        print("=" * 70)
        print("CONTEXT-BASED META-LEARNING (Multi-GPU DDP)")
        print("=" * 70)
        print(
            f"Nodes: {model.module.max_nodes if hasattr(model, 'module') else model.max_nodes}"
        )
        print(f"Total episodes per epoch: {episodes_per_epoch * world_size}")
        print(
            f"Context samples per episode: {k_context_range[0]}-{k_context_range[1]} (varies)"
        )
        print("  - k=0: Zero-shot learning")
        print("  - k≥1: Few-shot in-context learning")
        print(f"Query samples per episode: {n_queries}")
        print(
            f"Total labels per epoch: {episodes_per_epoch * world_size * n_queries:,}"
        )
        print(f"Learning Rate: {scaled_lr:.6f} (scaled from {lr})")
        print("=" * 70)

    for epoch in range(num_epochs):
        if rank == 0:
            print(
                f"\nEpoch {epoch+1}/{num_epochs}  [graph size: {min_nodes}–{max_nodes} nodes, uniform]"
            )

        # Fresh dataset for each epoch with unique seed per rank
        dataset = ContextQueryDataset(
            episodes_per_epoch=episodes_per_epoch,
            k_context_range=k_context_range,
            n_queries=n_queries,
            seed=42 + epoch + (rank * 10000),
            equation_type=equation_type,
            obs_range=(5, 500),
            int_range=(1, 200),
            n_nodes_range=(min_nodes, max_nodes),
            max_nodes=max_nodes,
            prior=prior,
            single_rc=single_rc,
        )

        # batch_size=None means the dataset yields one full episode at a time
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=4,
            pin_memory=True,
        )

        model.train()
        epoch_loss = 0
        epoch_acc = 0

        # Only show progress bar on the master rank
        loader_iter = (tqdm(train_loader, desc=f"Rank {rank}")
                       if rank == 0 else train_loader)

        for episode_idx, (
                context_obs,
                context_int,
                query_obs,
                query_int,
                query_labels,
                query_target_masks,
                context_target_masks,
                n_nodes,
        ) in enumerate(loader_iter):

            # Move data to device
            context_obs = context_obs.to(device)
            context_int = context_int.to(device)
            query_obs = query_obs.to(device)
            query_int = query_int.to(device)
            query_labels = query_labels.to(device)
            query_target_masks = query_target_masks.to(device)
            context_target_masks = context_target_masks.to(device)
            n_nodes = n_nodes.item()  # Convert to Python int

            # Training step with masking
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

            if rank == 0 and episode_idx % 10 == 0:
                loader_iter.set_postfix({
                    "loss": f"{loss:.4f}",
                    "acc": f"{acc:.4f}"
                })

        # --- EPOCH END ---
        avg_loss = epoch_loss / episodes_per_epoch
        avg_acc = epoch_acc / episodes_per_epoch

        if rank == 0:
            print(f"\n[Rank 0] Epoch {epoch+1} Finished")
            print(f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")

            # Save clean state_dict (stripping DDP's .module prefix)
            checkpoint_name = f"context_meta_multigpu_epoch_{epoch+1}.pt"
            torch.save(model.module.state_dict(), checkpoint_name)
            print(f"Saved checkpoint: {checkpoint_name}")

            # Real-data validation on PetShop (train split) and CausRCA
            run_real_data_validation(model, device, max_nodes, epoch)

    if rank == 0:
        print("\n✓ Context-based meta-learning complete!")
        torch.save(model.module.state_dict(), "meta_multigpu_final.pt")

    return model


def main():
    """Main multi-GPU context-based meta-learning"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-GPU context meta-learning")
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
        help=
        "Weight of the alarm-avoidance loss term (push the observed-symptom "
        "logit down; cf. finetune_causrca). Try 0.1-0.25 with --prior tabicl.",
    )
    parser.add_argument(
        "--single_rc",
        action="store_true",
        help="Force exactly one root cause per episode (--prior tabicl only; "
        "single-RC baseline for the multi-RC-training A/B).",
    )
    args = parser.parse_args()

    # Setup DDP
    rank, world_size, device = setup_ddp()

    if rank == 0:
        print(f"Initialized {world_size} GPUs")

    # Load model
    from prim.model.prim_neural_process import PRIMNeuralProcess

    model = PRIMNeuralProcess(
        embedding_size=160,
        num_attention_heads=8,
        mlp_hidden_size=512,
        num_layers=6,
        max_nodes=args.max_nodes,
        dropout=0.1,
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        if rank == 0:
            print(f"Resumed from checkpoint: {args.resume}")

    if rank == 0:
        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    # Train with context + queries approach
    model = train_context_meta_ddp(
        model,
        device,
        rank,
        world_size,
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

    cleanup_ddp()


if __name__ == "__main__":
    main()
