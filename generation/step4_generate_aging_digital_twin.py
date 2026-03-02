import os
import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path

def generate_aging_digital_twin():
    """
    Step 4: Aging-Aware Core Temperature Digital Twin (GPU Accelerated).
    Simulates HUGE amounts of data by processing 5 unique drive cycles 
    across 160 aging steps in massive parallel GPU batches.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "digital_twin_sets"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Starting Massive Parallel Aging Physics on {device}...")

    # NASA 18650 Battery Baseline Parameters
    capacity_ah_nom = 2.0  
    dt = 1.0  
    base_R0 = 0.05
    R1 = 0.009899; C1 = 15290.0
    R2 = 0.030116; C2 = 3236.0
    Rin = 3.0; Rout = 15.0
    Cc = 30.0; Cs = 15.0
    
    alpha1 = np.exp(-dt / (R1 * C1))
    alpha2 = np.exp(-dt / (R2 * C2))
    
    # --- Generate 5 distinct driving profiles (UDDS, HWFET, City, Commute, Aggressive) ---
    T_steps = 1500 # 1500 seconds per drive
    time_s = np.arange(0, T_steps, dt)
    
    profiles = [
        10 * np.sin(time_s / 20) + 5 * np.cos(time_s / 5), # Mixed
        15 * np.sin(time_s / 10),                            # Aggressive
        8 * np.sin(time_s / 30) + 2,                         # Highway
        5 * np.cos(time_s / 15) + 3 * np.sin(time_s / 5),    # City Stop-Go
        12 * np.sin(time_s / 25) + 4                         # Commute
    ]
    
    I_profiles = torch.tensor(np.array(profiles), dtype=torch.float32, device=device) # [5 profiles, 1500 steps]
    num_profiles = I_profiles.shape[0]
    num_cycles = 160
    
    # We will compute 160 cycles * 5 profiles = 800 Scenarios all at the EXACT SAME TIME on the GPU
    batch_size = num_cycles * num_profiles
    
    # Create Batched SOH and Resistance (R0) parameters
    cycle_idx = torch.arange(1, num_cycles + 1, device=device).float()
    soh_k = torch.clamp(1.0 - (cycle_idx / 300.0) ** 1.1, min=0.70)
    current_r0 = base_R0 + 0.15 * (1.0 - soh_k)
    q_nom = (capacity_ah_nom * soh_k) * 3600.0
    
    # Expand to full batch [800]
    soh_batch = soh_k.repeat_interleave(num_profiles)
    r0_batch = current_r0.repeat_interleave(num_profiles)
    q_nom_batch = q_nom.repeat_interleave(num_profiles)
    cycle_batch = cycle_idx.repeat_interleave(num_profiles)
    
    # Expand drive profiles to batch [800, 1500]
    I_drive = I_profiles.repeat(num_cycles, 1)
    
    # Physics State Tensors [800]
    soc = torch.ones(batch_size, device=device)
    V1 = torch.zeros(batch_size, device=device)
    V2 = torch.zeros(batch_size, device=device)
    tc = torch.full((batch_size,), 25.0, device=device)
    ts = torch.full((batch_size,), 25.0, device=device)
    tamb = 25.0
    
    # Storage Tensors
    v_term_all = torch.zeros(batch_size, T_steps, device=device)
    tc_all = torch.zeros(batch_size, T_steps, device=device)
    ts_all = torch.zeros(batch_size, T_steps, device=device)
    
    print(f"⚡ Firing {batch_size} parallel physics simulations ({batch_size * T_steps} tensor operations) via PyTorch...")
    start_time = time.time()
    
    for i in range(T_steps):
        current = I_drive[:, i]
        
        # --- 2-RC ECM Vectorized Math ---
        v_ocv = 3.2 + 1.0 * soc  
        v_term = v_ocv - (current * r0_batch) - V1 - V2
        
        q_gen = torch.abs(current * (v_ocv - v_term)) 
        
        # --- 2-State EETM Vectorized Math ---
        dtc = (q_gen - (tc - ts) / Rin) / Cc
        dts = ((tc - ts) / Rin - (ts - tamb) / Rout) / Cs
        
        tc += dtc * dt
        ts += dts * dt
        
        soc -= (current * dt) / q_nom_batch
        soc = torch.clamp(soc, 0.0, 1.0)
        
        V1 = alpha1 * V1 + current * R1 * (1.0 - alpha1)
        V2 = alpha2 * V2 + current * R2 * (1.0 - alpha2)
        
        # Record
        v_term_all[:, i] = v_term
        tc_all[:, i] = tc
        ts_all[:, i] = ts

    total_hours = (batch_size * T_steps) / 3600
    print(f"⏱️ GPU Simulation finished in {time.time() - start_time:.2f} seconds!")
    print(f"💥 Successfully synthesized {total_hours:.1f} hours of physics Ground Truth.")
    
    # Move huge payload to CPU to save to CSV
    print("💾 Formatting to CSV (this may take a few moments)...")
    I_drive_cpu = I_drive.cpu().numpy()
    v_term_cpu = v_term_all.cpu().numpy()
    tc_cpu = tc_all.cpu().numpy()
    ts_cpu = ts_all.cpu().numpy()
    r0_cpu = r0_batch.cpu().numpy()
    soh_cpu = soh_batch.cpu().numpy()
    cycle_cpu = cycle_batch.cpu().numpy()
    
    records = []
    # Build records efficiently
    for b in range(batch_size):
        for t in range(T_steps):
            records.append({
                "time_s": time_s[t],
                "cycle_age": cycle_cpu[b],
                "soh_true": soh_cpu[b],
                "r0_ohms": r0_cpu[b],
                "current_A": I_drive_cpu[b, t],
                "voltage_V": v_term_cpu[b, t],
                "temp_surface_C": ts_cpu[b, t],
                "temp_core_C_TARGET": tc_cpu[b, t]
            })
            
    df = pd.DataFrame(records)
    out_path = data_dir / "augmented_aging_twin_dataset.csv"
    df.to_csv(out_path, index=False)
    
    print(f"✅ Extracted ~{len(records)} rows to {out_path.name}")

if __name__ == "__main__":
    generate_aging_digital_twin()
