# generate_data.py
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from tqdm import tqdm
import argparse
import random
import time
import matplotlib.pyplot as plt
import copy
import sys
import json
import pandas as pd


def load_network():
    return pn.case30()


def sample_loads(
    net,
    load_scale_range=(0.8, 1.2),
    gen_perturbation=0.1,
    gen_vm_perturb_range=(0.9, 1.1),
    add_sgen_prob=0.08,
    sgen_q_range=(-0.5, 0.8)
):
    
    for _, load in net.load.iterrows():
        scale_factor = random.uniform(*load_scale_range)
        net.load.at[load.name, 'p_mw'] *= scale_factor
        net.load.at[load.name, 'q_mvar'] *= scale_factor

    for _, gen in net.gen.iterrows():
        perturb_p = random.uniform(1 - gen_perturbation, 1 + gen_perturbation)
        net.gen.at[gen.name, 'p_mw'] *= perturb_p

        if 'vm_pu' in net.gen.columns:
            net.gen.at[gen.name, 'vm_pu'] = random.uniform(*gen_vm_perturb_range)

    for b in net.bus.index:
        if random.random() < add_sgen_prob:
            q = random.uniform(*sgen_q_range)
            pp.create_sgen(net, bus=int(b), p_mw=0.0, q_mvar=q)


def run_powerflow(net):
   
    try:
        pp.runpp(net, algorithm='nr')
        return net.converged
    except Exception:
        return False


def extract_bus_data(net):
   
    bus_data = []
    for _, bus in net.bus.iterrows():
        p = net.res_bus.loc[bus.name, 'p_mw'] / net.sn_mva
        q = net.res_bus.loc[bus.name, 'q_mvar'] / net.sn_mva
        v_mag = net.res_bus.loc[bus.name, 'vm_pu']
        v_ang = net.res_bus.loc[bus.name, 'va_degree']
        bus_data.append([p, q, v_mag, v_ang])
    return np.array(bus_data)


def compute_specified_injections(net):
    p_spec = np.zeros(len(net.bus))
    q_spec = np.zeros(len(net.bus))

    for _, load in net.load.iterrows():
        p_spec[load.bus] -= load.p_mw
        q_spec[load.bus] -= load.q_mvar

    for _, sgen in net.sgen.iterrows():
        p_spec[sgen.bus] += sgen.p_mw
        q_spec[sgen.bus] += sgen.q_mvar if "q_mvar" in net.sgen.columns else 0

    for _, gen in net.gen.iterrows():
        p_spec[gen.bus] += gen.p_mw
        q_spec[gen.bus] += gen.q_mvar if "q_mvar" in net.gen.columns else 0

    return np.column_stack((p_spec / net.sn_mva, q_spec / net.sn_mva))


def prepare_tokens(dataset, net):
    num_samples, num_buses, ncols = dataset.shape
    assert ncols >= 4

    bus_type_raw = net.bus['type']
    if np.issubdtype(bus_type_raw.dtype, np.integer):
        unique_types = np.unique(bus_type_raw)
        preferred = {0: 'PQ', 1: 'PV', 2: 'SLACK'}
        type_names = []
        for t in sorted(unique_types):
            type_names.append(preferred.get(int(t), f"TYPE_{int(t)}"))
        type_to_idx = {t: i for i, t in enumerate(type_names)}
        bus_type_idx = np.array([type_to_idx[preferred.get(int(x), f"TYPE_{int(x)}")] for x in bus_type_raw])
    else:
        bus_type_str = bus_type_raw.astype(str).str.lower()
        def map_str_to_label(s):
            if 'pv' in s: return 'PV'
            if 'pq' in s: return 'PQ'
            if any(k in s for k in ('slack', 'swing', 'ref')): return 'SLACK'
            return s.strip().upper()

        labels = bus_type_str.apply(map_str_to_label).tolist()
        unique_labels = []
        for lab in labels:
            if lab not in unique_labels: unique_labels.append(lab)
        ordered = []
        for want in ['PQ', 'PV', 'SLACK']:
            if want in unique_labels: ordered.append(want)
        for lab in unique_labels:
            if lab not in ordered: ordered.append(lab)
        type_to_idx = {lab: idx for idx, lab in enumerate(ordered)}
        bus_type_idx = np.array([type_to_idx[map_str_to_label(s)] for s in bus_type_raw.astype(str)])

    num_type_dims = len(type_to_idx)
    token_dim = 2 + num_type_dims  

    X_tokens = np.zeros((num_samples, num_buses, token_dim), dtype=float)
    Y_targets = np.zeros((num_samples, num_buses, 2), dtype=float)

    bus_type_one_hot = np.zeros((num_buses, num_type_dims), dtype=float)
    bus_type_one_hot[np.arange(num_buses), bus_type_idx] = 1.0

    for i in range(num_samples):
        sample = dataset[i]  
        P_spec = sample[:, 0]
        Q_spec = sample[:, 1]
        Vmag = sample[:, 2]
        Vang = sample[:, 3]

        X_sample = np.concatenate([ P_spec[:, None], Q_spec[:, None], bus_type_one_hot ], axis=1)
        Y_sample = np.stack([Vmag, Vang], axis=1)

        X_tokens[i] = X_sample
        Y_targets[i] = Y_sample

    return X_tokens, Y_targets


def generate_data(num_samples, output_path, seed=None, angle_radians=False):
    if seed is not None:
        random.seed(seed)

    base_net = load_network()
    dataset = []
    converged_count = 0
    fail_reasons = []
    
    # UPDATED: Added list to store solver times for statistics
    solver_times = []

    for _ in tqdm(range(num_samples), desc="Generating data"):
        net = copy.deepcopy(base_net)
        sample_loads(net, load_scale_range=(0.6, 1.6), gen_perturbation=0.5)

        start_time = time.time()
        try:
            if run_powerflow(net):
                solver_time = time.time() - start_time
                solver_times.append(solver_time) # UPDATED: Store time for each converged run

                p_q_spec = compute_specified_injections(net)
                v_mag = net.res_bus["vm_pu"].values
                va = net.res_bus["va_degree"].values
                if angle_radians:
                    va = np.deg2rad(va)

                bus_data = np.column_stack((p_q_spec, v_mag, va))
                dataset.append(bus_data)
                converged_count += 1
            else:
                fail_reasons.append("Power flow did not converge.")
        except Exception as e:
            fail_reasons.append(str(e))

    dataset = np.array(dataset)

    # Prepare tokens and save everything into the .npz file
    X_tokens, Y_targets = prepare_tokens(dataset, base_net)
    np.savez_compressed(output_path, X_tokens=X_tokens, Y_targets=Y_targets)
    
    print(f"\n--- Data Generation Summary ---")
    print(f"Converged samples: {converged_count}/{num_samples}")
    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {dataset.shape}")

    # UPDATED: Calculate and print detailed solver time statistics
    if solver_times:
        print(f"\n--- NR Solver Time Statistics (for converged runs) ---")
        print(f"  - Average: {np.mean(solver_times) * 1000:.2f} ms")
        print(f"  - Std Dev: {np.std(solver_times) * 1000:.2f} ms")
        print(f"  - Min:     {np.min(solver_times) * 1000:.2f} ms")
        print(f"  - Max:     {np.max(solver_times) * 1000:.2f} ms")

    if fail_reasons:
        print("\n--- Failure Diagnostics ---")
        for reason in fail_reasons[:5]:
            print(f"- {reason}")
        if len(fail_reasons) > 5:
            print(f"...and {len(fail_reasons) - 5} more.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate power flow dataset for IEEE-30 bus system.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for reproducibility.")
    parser.add_argument("--angle_radians", action="store_true", help="Store voltage angles in radians.")
    parser.add_argument("--output_path", type=str, default="dataset_ieee30.npz", help="Output path for dataset file.")

    # Handle running in Jupyter/IPython environments
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args(args=[]) 
    else:
        args = parser.parse_args()

    generate_data(args.num_samples, args.output_path, args.seed, args.angle_radians)