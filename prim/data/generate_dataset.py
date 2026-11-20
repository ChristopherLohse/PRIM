"""
Generate Dataset of 10,000 Observational/Interventional Pairs

Each sample contains:
- 300 observational samples
- 100 interventional samples
- Binary label vector indicating which node was intervened on
"""

import numpy as np
import pickle
import random
from prim.data.dag_generator_simplified import (
    generate_observational_and_interventional_data,
    load_nn_config,
)


def print_progress(current, total, prefix="Progress"):
    """Simple progress printer."""
    if current % max(1, total // 20) == 0 or current == total - 1:
        percentage = (current + 1) / total * 100
        print(f"{prefix}: {current + 1}/{total} ({percentage:.1f}%)")


def generate_intervention_dataset(
    n_samples: int = 200000,
    n_nodes_range: tuple = (5, 10),
    n_obs: int = 300,
    n_int: int = 100,
    max_nodes: int = 10,  # Maximum K for PFN-style encoding
    equation_type: str = "mixed",  # 'linear', 'nn', or 'mixed' (50/50)
    graph_type: str = "erdos-renyi",
    edge_probability_range: tuple = (0.3, 0.5),
    noise_scale_range: tuple = (0.3, 0.7),
    weight_change_factor_range: tuple = (2.5, 4.5),
    flip_signs_prob: float = 0.7,
    do_intervention_prob:
    float = 0.0,  # fraction using hard do-intervention (stuck sensor)
    stuck_magnitude_range: tuple = (2.0,
                                    4.0),  # std-dev offset for stuck value
    temporal_autocorr_range: tuple = (0.0, 0.0),  # AR(1) range; (0,0) = i.i.d.
    save_path: str = "intervention_dataset.pkl",
    seed: int = 42,
    nn_config_path=None,
):
    """
    Generate a dataset with PFN-style flexible encoding.

    Following the PFN approach:
    - Sample number of dimensions k uniformly from n_nodes_range
    - Zero-pad to max_nodes (K)
    - Scale features by K/k to preserve magnitude

    Args:
        n_samples: Number of DAG instances to generate
        n_nodes_range: Range of nodes per DAG (min, max), e.g., (10, 20)
        n_obs: Number of observational samples per instance
        n_int: Number of interventional samples per instance
        max_nodes: Maximum number of nodes K (for padding)
        equation_type: 'linear', 'nonlinear', or 'nn' or 'mixed'
        graph_type: Type of DAG
        edge_probability_range: Range of edge probabilities (min, max)
        noise_scale_range: Range of Gaussian noise scales (min, max)
        weight_change_factor_range: Range of intervention strengths (min, max)
        flip_signs_prob: Probability of flipping signs during weight-change intervention
        do_intervention_prob: Fraction of samples using hard do-intervention (stuck
                              sensor). The remaining fraction uses soft weight-change.
                              Set to 0.5 to match causRCA-style "stuck sensor" faults.
        stuck_magnitude_range: (min, max) z-score offset for the stuck constant value.
        temporal_autocorr_range: (min, max) AR(1) autocorrelation for noise per sample.
                                 Set to (0.8, 0.99) to mimic slowly-changing time-series.
        save_path: Path to save the dataset
        seed: Base random seed

    Returns:
        dataset: Dictionary containing all data
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load NN config if needed
    nn_config = None
    if equation_type in ("nn", "mixed"):
        nn_config = load_nn_config(nn_config_path)

    dataset = {
        "observational_data":
        [],  # List of (n_obs, max_nodes) arrays - PADDED & SCALED
        "interventional_data":
        [],  # List of (n_int, max_nodes) arrays - PADDED & SCALED
        "intervention_labels":
        [],  # List of (max_nodes,) binary vectors - PADDED
        "intervention_nodes": [],  # List of intervention node indices
        "dags": [],  # List of DAG edge lists
        "n_nodes_list": [],  # List of actual number of nodes per sample
        "edge_probabilities": [],  # List of edge probabilities used
        "noise_scales": [],  # List of noise scales used
        "weight_change_factors": [],  # List of weight change factors used
        "flip_signs_list": [],  # List of whether signs were flipped
        "scaling_factors": [],  # List of K/k scaling factors applied
        "equation_types":
        [],  # List of equation types per sample ('linear' or 'nn')
        "metadata": {
            "n_samples":
            n_samples,
            "n_nodes_range":
            n_nodes_range,
            "max_nodes":
            max_nodes,
            "n_obs":
            n_obs,
            "n_int":
            n_int,
            "equation_type":
            equation_type,
            "graph_type":
            graph_type,
            "edge_probability_range":
            edge_probability_range,
            "noise_scale_range":
            noise_scale_range,
            "weight_change_factor_range":
            weight_change_factor_range,
            "flip_signs_prob":
            flip_signs_prob,
            "do_intervention_prob":
            do_intervention_prob,
            "stuck_magnitude_range":
            stuck_magnitude_range,
            "temporal_autocorr_range":
            temporal_autocorr_range,
            "encoding":
            "PFN-style: zero-padded and scaled by K/k",
            "seed":
            seed,
            "nn_config_path":
            nn_config_path if equation_type in ("nn", "mixed") else None,
        },
    }

    print(
        f"Generating {n_samples:,} samples with PFN-style flexible encoding..."
    )
    print("Configuration:")
    print(
        f"  - Nodes per DAG: {n_nodes_range[0]}-{n_nodes_range[1]} (uniformly sampled)"
    )
    print(f"  - Max nodes (K): {max_nodes}")
    print(f"  - Observational samples: {n_obs}")
    print(f"  - Interventional samples: {n_int}")
    print(f"  - Equation type: {equation_type}")
    print(f"  - Edge probability range: {edge_probability_range}")
    print(f"  - Weight change factor range: {weight_change_factor_range}")
    print(
        f"  - Do-intervention (stuck sensor) prob: {do_intervention_prob:.0%}")
    print(f"  - Stuck magnitude range (std devs): {stuck_magnitude_range}")
    print(f"  - Temporal autocorr range: {temporal_autocorr_range}")
    print(f"  - Encoding: Zero-padded to K={max_nodes}, scaled by K/k")
    print()

    for i in range(n_samples):
        # Print progress
        print_progress(i, n_samples, "Generating samples")

        # Use different seed for each sample
        sample_seed = seed + i
        np.random.seed(sample_seed)
        random.seed(sample_seed)

        # Randomly sample parameters for this instance
        n_nodes = np.random.randint(n_nodes_range[0],
                                    n_nodes_range[1] + 1)  # k
        edge_probability = np.random.uniform(edge_probability_range[0],
                                             edge_probability_range[1])
        noise_scale = np.random.uniform(noise_scale_range[0],
                                        noise_scale_range[1])
        weight_change_factor = np.random.uniform(weight_change_factor_range[0],
                                                 weight_change_factor_range[1])
        flip_signs = np.random.random() < flip_signs_prob
        use_do_intervention = np.random.random() < do_intervention_prob
        temporal_autocorr = np.random.uniform(temporal_autocorr_range[0],
                                              temporal_autocorr_range[1])

        # Pick equation type for this sample
        if equation_type == "mixed":
            sample_eq_type = np.random.choice(["linear", "nn", "gp"])
        else:
            sample_eq_type = equation_type

        # PFN-style scaling factor: K/k
        scaling_factor = max_nodes / n_nodes

        try:
            # Generate data with sampled parameters
            obs_data, int_data, dag, int_node = (
                generate_observational_and_interventional_data(
                    n_nodes=n_nodes,
                    n_obs_samples=n_obs,
                    n_int_samples=n_int,
                    intervention_node=None,  # Random selection
                    equation_type=sample_eq_type,
                    graph_type=graph_type,
                    edge_probability=edge_probability,
                    noise_scale=noise_scale,
                    weight_change_factor=weight_change_factor,
                    flip_signs=flip_signs,
                    seed=sample_seed,
                    nn_config=nn_config if sample_eq_type == "nn" else None,
                    do_intervention=use_do_intervention,
                    stuck_magnitude_range=stuck_magnitude_range,
                    temporal_autocorr=temporal_autocorr,
                ))

            # Apply PFN-style encoding: normalize, zero-pad and scale
            # CRITICAL: Normalize using ONLY observational statistics
            # This preserves the mean shift caused by intervention

            # Compute normalization parameters from observational data only
            obs_mean = obs_data.mean(axis=0, keepdims=True)
            obs_std = obs_data.std(axis=0, keepdims=True) + 1e-8

            # Normalize BOTH datasets using observational statistics
            obs_data_normalized = (obs_data - obs_mean) / obs_std
            int_data_normalized = (int_data -
                                   obs_mean) / obs_std  # Use obs stats!

            if n_nodes < max_nodes:
                # Zero-pad to max_nodes
                obs_data_padded = np.pad(
                    obs_data_normalized,
                    ((0, 0), (0, max_nodes - n_nodes)),
                    mode="constant",
                    constant_values=0,
                )
                int_data_padded = np.pad(
                    int_data_normalized,
                    ((0, 0), (0, max_nodes - n_nodes)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                obs_data_padded = obs_data_normalized
                int_data_padded = int_data_normalized

            # Scale by K/k to preserve magnitude (PFN-style)
            obs_data_scaled = obs_data_padded * scaling_factor
            int_data_scaled = int_data_padded * scaling_factor

            # Create intervention label vector (one-hot encoding, padded to max_nodes)
            intervention_label = np.zeros(max_nodes, dtype=np.int32)
            intervention_label[int_node] = 1

            # Store data
            dataset["observational_data"].append(obs_data_scaled)
            dataset["interventional_data"].append(int_data_scaled)
            dataset["intervention_labels"].append(intervention_label)
            dataset["intervention_nodes"].append(int_node)
            dataset["dags"].append(list(dag.edges()))
            dataset["n_nodes_list"].append(n_nodes)
            dataset["edge_probabilities"].append(edge_probability)
            dataset["noise_scales"].append(noise_scale)
            dataset["weight_change_factors"].append(weight_change_factor)
            dataset["flip_signs_list"].append(flip_signs)
            dataset["scaling_factors"].append(scaling_factor)
            dataset["equation_types"].append(sample_eq_type)

        except Exception as e:
            print(f"\nWarning: Failed to generate sample {i}: {e}")
            print("Retrying with different seed...")
            # Retry with perturbed seed
            sample_seed = seed + i + n_samples
            np.random.seed(sample_seed)
            random.seed(sample_seed)

            try:
                obs_data, int_data, dag, int_node = (
                    generate_observational_and_interventional_data(
                        n_nodes=n_nodes,
                        n_obs_samples=n_obs,
                        n_int_samples=n_int,
                        intervention_node=None,
                        equation_type=sample_eq_type,
                        graph_type=graph_type,
                        edge_probability=edge_probability,
                        noise_scale=noise_scale,
                        weight_change_factor=weight_change_factor,
                        flip_signs=flip_signs,
                        seed=sample_seed,
                        nn_config=nn_config
                        if sample_eq_type == "nn" else None,
                        do_intervention=use_do_intervention,
                        stuck_magnitude_range=stuck_magnitude_range,
                        temporal_autocorr=temporal_autocorr,
                    ))

                # Apply PFN-style encoding
                # Normalize using ONLY observational statistics
                obs_mean = obs_data.mean(axis=0, keepdims=True)
                obs_std = obs_data.std(axis=0, keepdims=True) + 1e-8

                obs_data_normalized = (obs_data - obs_mean) / obs_std
                int_data_normalized = (int_data -
                                       obs_mean) / obs_std  # Use obs stats!

                if n_nodes < max_nodes:
                    obs_data_padded = np.pad(
                        obs_data_normalized,
                        ((0, 0), (0, max_nodes - n_nodes)),
                        mode="constant",
                        constant_values=0,
                    )
                    int_data_padded = np.pad(
                        int_data_normalized,
                        ((0, 0), (0, max_nodes - n_nodes)),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    obs_data_padded = obs_data_normalized
                    int_data_padded = int_data_normalized

                obs_data_scaled = obs_data_padded * scaling_factor
                int_data_scaled = int_data_padded * scaling_factor

                intervention_label = np.zeros(max_nodes, dtype=np.int32)
                intervention_label[int_node] = 1

                dataset["observational_data"].append(obs_data_scaled)
                dataset["interventional_data"].append(int_data_scaled)
                dataset["intervention_labels"].append(intervention_label)
                dataset["intervention_nodes"].append(int_node)
                dataset["dags"].append(list(dag.edges()))
                dataset["n_nodes_list"].append(n_nodes)
                dataset["edge_probabilities"].append(edge_probability)
                dataset["noise_scales"].append(noise_scale)
                dataset["weight_change_factors"].append(weight_change_factor)
                dataset["flip_signs_list"].append(flip_signs)
                dataset["scaling_factors"].append(scaling_factor)
                dataset["equation_types"].append(sample_eq_type)

            except Exception as e2:
                print(f"Failed again: {e2}. Skipping sample {i}.")
                continue

    # Convert lists to arrays
    dataset["observational_data"] = np.array(
        dataset["observational_data"])  # (n_samples, n_obs, max_nodes)
    dataset["interventional_data"] = np.array(
        dataset["interventional_data"])  # (n_samples, n_int, max_nodes)
    dataset["intervention_labels"] = np.array(
        dataset["intervention_labels"])  # (n_samples, max_nodes)
    dataset["intervention_nodes"] = np.array(dataset["intervention_nodes"])
    dataset["n_nodes_list"] = np.array(dataset["n_nodes_list"])
    dataset["edge_probabilities"] = np.array(dataset["edge_probabilities"])
    dataset["noise_scales"] = np.array(dataset["noise_scales"])
    dataset["weight_change_factors"] = np.array(
        dataset["weight_change_factors"])
    dataset["flip_signs_list"] = np.array(dataset["flip_signs_list"])
    dataset["scaling_factors"] = np.array(dataset["scaling_factors"])
    dataset["equation_types"] = np.array(dataset["equation_types"])

    print("\n✓ Dataset generation complete!")
    print(f"\nDataset size: {len(dataset['intervention_nodes'])} samples")
    print("\nPFN-style encoding applied:")
    print(f"  All data padded to: {max_nodes} nodes")
    print(
        f"  Scaling factors (K/k): min={dataset['scaling_factors'].min():.3f}, "
        f"max={dataset['scaling_factors'].max():.3f}, mean={dataset['scaling_factors'].mean():.3f}"
    )

    print("\nNode distribution (original k before padding):")
    for n in range(n_nodes_range[0], n_nodes_range[1] + 1):
        count = np.sum(dataset["n_nodes_list"] == n)
        percentage = (count / len(dataset["n_nodes_list"])) * 100
        scaling = max_nodes / n
        print(
            f"  {n} nodes: {count:5d} ({percentage:5.2f}%) - scaled by {scaling:.3f}"
        )

    print(f"\nData shapes (all padded to K={max_nodes}):")
    print(f"  Observational: {dataset['observational_data'].shape}")
    print(f"  Interventional: {dataset['interventional_data'].shape}")
    print(f"  Labels: {dataset['intervention_labels'].shape}")

    print(
        f"\nEdge probability: {dataset['edge_probabilities'].mean():.3f} ± {dataset['edge_probabilities'].std():.3f}"
    )
    print(
        f"Noise scale: {dataset['noise_scales'].mean():.3f} ± {dataset['noise_scales'].std():.3f}"
    )
    print(
        f"Weight change factor: {dataset['weight_change_factors'].mean():.3f} ± {dataset['weight_change_factors'].std():.3f}"
    )
    print(
        f"Sign flips: {dataset['flip_signs_list'].sum()} / {len(dataset['flip_signs_list'])} ({dataset['flip_signs_list'].mean()*100:.1f}%)"
    )

    # Save dataset
    print(f"\nSaving dataset to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print("✓ Dataset saved successfully!")
    print(f"  File size: {get_file_size_mb(save_path):.2f} MB")

    return dataset


def load_dataset(path: str):
    """Load a saved dataset."""
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    import os

    return os.path.getsize(path) / (1024 * 1024)


def print_dataset_info(dataset):
    """Print information about a dataset."""
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nDataset size:")
    print(f"  Total samples: {len(dataset['intervention_nodes'])}")

    print("\nMetadata:")
    for key, value in dataset["metadata"].items():
        print(f"  {key}: {value}")

    print("\nNode size distribution:")
    n_nodes_list = dataset.get("n_nodes_list", None)
    if n_nodes_list is not None:
        unique_sizes = np.unique(n_nodes_list)
        for size in unique_sizes:
            count = np.sum(n_nodes_list == size)
            percentage = (count / len(n_nodes_list)) * 100
            print(f"  {size} nodes: {count:5d} samples ({percentage:5.2f}%)")

    print("\nParameter distributions:")
    if "edge_probabilities" in dataset:
        print(
            f"  Edge probability: {dataset['edge_probabilities'].mean():.3f} ± {dataset['edge_probabilities'].std():.3f}"
        )
    if "noise_scales" in dataset:
        print(
            f"  Noise scale: {dataset['noise_scales'].mean():.3f} ± {dataset['noise_scales'].std():.3f}"
        )
    if "weight_change_factors" in dataset:
        print(
            f"  Weight change factor: {dataset['weight_change_factors'].mean():.3f} ± {dataset['weight_change_factors'].std():.3f}"
        )
    if "flip_signs_list" in dataset:
        flip_count = dataset["flip_signs_list"].sum()
        flip_pct = dataset["flip_signs_list"].mean() * 100
        print(
            f"  Sign flips: {flip_count} / {len(dataset['flip_signs_list'])} ({flip_pct:.1f}%)"
        )

    print("\nExample sample (index 0):")
    print(f"  Intervention node: {dataset['intervention_nodes'][0]}")
    print(f"  Intervention label: {dataset['intervention_labels'][0]}")
    print(f"  Number of nodes: {len(dataset['intervention_labels'][0])}")
    print(f"  DAG edges: {dataset['dags'][0]}")
    print(
        f"  Observational data shape: {dataset['observational_data'][0].shape}"
    )
    print(
        f"  Interventional data shape: {dataset['interventional_data'][0].shape}"
    )
    if "edge_probabilities" in dataset:
        print(f"  Edge probability: {dataset['edge_probabilities'][0]:.3f}")
    if "weight_change_factors" in dataset:
        print(
            f"  Weight change factor: {dataset['weight_change_factors'][0]:.3f}"
        )

    print("\n" + "=" * 70)


def create_train_val_test_split(dataset,
                                train_ratio=0.7,
                                val_ratio=0.15,
                                seed=42):
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: The full dataset
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        seed: Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    np.random.seed(seed)

    n_samples = len(dataset["intervention_nodes"])
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def create_split(idx):
        split = {
            # Now these are numpy arrays (PFN-style: fixed size)
            "observational_data": dataset["observational_data"][idx],
            "interventional_data": dataset["interventional_data"][idx],
            "intervention_labels": dataset["intervention_labels"][idx],
            "intervention_nodes": dataset["intervention_nodes"][idx],
            "dags": [dataset["dags"][i] for i in idx],
            "metadata": dataset["metadata"].copy(),
        }

        # Add optional fields if they exist
        if "n_nodes_list" in dataset:
            split["n_nodes_list"] = dataset["n_nodes_list"][idx]
        if "edge_probabilities" in dataset:
            split["edge_probabilities"] = dataset["edge_probabilities"][idx]
        if "noise_scales" in dataset:
            split["noise_scales"] = dataset["noise_scales"][idx]
        if "weight_change_factors" in dataset:
            split["weight_change_factors"] = dataset["weight_change_factors"][
                idx]
        if "flip_signs_list" in dataset:
            split["flip_signs_list"] = dataset["flip_signs_list"][idx]
        if "scaling_factors" in dataset:
            split["scaling_factors"] = dataset["scaling_factors"][idx]
        if "equation_types" in dataset:
            split["equation_types"] = dataset["equation_types"][idx]

        return split

    train_dataset = create_split(train_idx)
    val_dataset = create_split(val_idx)
    test_dataset = create_split(test_idx)

    print("\nDataset split:")
    print(
        f"  Training:   {len(train_idx):5d} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_idx):5d} samples ({val_ratio*100:.1f}%)")
    print(
        f"  Test:       {len(test_idx):5d} samples ({(1-train_ratio-val_ratio)*100:.1f}%)"
    )

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Generate the full dataset with PFN-style flexible encoding.
    # Parameters tuned to match real-world monitoring data (e.g. causRCA digital twin):
    #   - n_nodes_range extended to (10, 20) to match 11-17 node subgraphs
    #   - do_intervention_prob=0.5: half samples are "stuck sensor" (zero variance fault)
    #   - temporal_autocorr_range=(0.8, 0.99): slow time-series dynamics
    dataset = generate_intervention_dataset(
        n_samples=120000,
        n_nodes_range=(10,
                       20),  # Extended to cover causRCA's 11-17 node subgraphs
        max_nodes=50,  # Must match model architecture's max_nodes
        n_obs=300,
        n_int=100,
        equation_type="mixed",
        graph_type="erdos-renyi",
        edge_probability_range=(0.3, 0.5),
        noise_scale_range=(0.1, 0.2),
        weight_change_factor_range=(3, 4.5),
        flip_signs_prob=0.7,
        do_intervention_prob=0.5,  # 50% hard do-interventions (stuck sensor)
        stuck_magnitude_range=(
            2.0, 4.0),  # Stuck value 2-4 std devs from normal mean
        temporal_autocorr_range=(0.8, 0.99),  # Slow time-series dynamics
        save_path="intervention_dataset_10k.pkl",
        seed=42,
    )

    # Print dataset information
    print_dataset_info(dataset)

    # Create train/val/test splits
    print("\n" + "=" * 70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 70)

    train_data, val_data, test_data = create_train_val_test_split(
        dataset, train_ratio=0.7, val_ratio=0.15, seed=42)

    # Save splits
    print("\nSaving splits...")
    with open("train_dataset.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("val_dataset.pkl", "wb") as f:
        pickle.dump(val_data, f)
    with open("test_dataset.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print(f"✓ Train set saved: {get_file_size_mb('train_dataset.pkl'):.2f} MB")
    print(f"✓ Val set saved: {get_file_size_mb('val_dataset.pkl'):.2f} MB")
    print(f"✓ Test set saved: {get_file_size_mb('test_dataset.pkl'):.2f} MB")

    # Example: Load and inspect a sample
    print("\n" + "=" * 70)
    print("EXAMPLE SAMPLE INSPECTION")
    print("=" * 70)

    sample_idx = 0
    print(f"\nSample {sample_idx}:")
    print(f"  Number of nodes: {dataset['n_nodes_list'][sample_idx]}")
    print(f"  Intervention node: {dataset['intervention_nodes'][sample_idx]}")
    print(
        f"  Intervention label: {dataset['intervention_labels'][sample_idx]}")
    print(f"  DAG edges: {dataset['dags'][sample_idx]}")
    print(
        f"  Edge probability: {dataset['edge_probabilities'][sample_idx]:.3f}")
    print(f"  Noise scale: {dataset['noise_scales'][sample_idx]:.3f}")
    print(
        f"  Weight change factor: {dataset['weight_change_factors'][sample_idx]:.3f}"
    )
    print(f"  Signs flipped: {dataset['flip_signs_list'][sample_idx]}")
    print(
        f"\n  Observational data (first 5 rows):\n{dataset['observational_data'][sample_idx][:5]}"
    )
    print(
        f"\n  Interventional data (first 5 rows):\n{dataset['interventional_data'][sample_idx][:5]}"
    )

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print("\nFiles created:")
    print("  - intervention_dataset_10k.pkl (full dataset)")
    print("  - train_dataset.pkl (70% of data)")
    print("  - val_dataset.pkl (15% of data)")
    print("  - test_dataset.pkl (15% of data)")
    print("\nDataset features:")
    print("  - Variable node sizes (5-10 nodes per DAG)")
    print("  - Variable edge probabilities (0.3-0.5)")
    print("  - Variable intervention strengths (2.5-4.5x)")
    print("  - Variable noise scales (0.3-0.7)")
    print("  - Mixed sign flipping (70% probability)")
    print("\nYou can load these files using:")
    print("  dataset = load_dataset('intervention_dataset_10k.pkl')")
    print("=" * 70)
