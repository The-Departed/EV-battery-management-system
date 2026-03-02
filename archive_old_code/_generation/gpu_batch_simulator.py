"""
GPU-Batched Physics Simulator
==============================
Reimplements the 2-RC ECM + 2nd-Order EETM using PyTorch tensors.

Key idea: instead of simulating one scenario at a time, we simulate ALL
scenarios simultaneously as a batched tensor operation on GPU.

  CPU (old):  for each scenario: simulate()  → O(N * T) sequential
  GPU (new):  simulate_batch(all_scenarios)  → O(T) parallel across N

Physics equations (Euler integration, same as batch_simulator.py):
  ECM (2-RC):
      SOC[k+1]  = SOC[k] - I[k]*dt / (3600*Q_nom)
      OCV[k]    = a + b * SOC[k]          (linear OCV model)
      V1[k+1]   = V1[k] * exp(-dt/(R1*C1)) + I[k]*R1*(1 - exp(-dt/(R1*C1)))
      V2[k+1]   = V2[k] * exp(-dt/(R2*C2)) + I[k]*R2*(1 - exp(-dt/(R2*C2)))
      V[k]      = OCV[k] - I[k]*R0 - V1[k] - V2[k]
      Q_gen[k]  = I[k] * (OCV[k] - V[k])

  EETM (2-state):
      Tc[k+1] = Tc[k] + dt/Cc * (Q_gen[k] - (Tc[k]-Ts[k])/Rin)
      Ts[k+1] = Ts[k] + dt/Cs * ((Tc[k]-Ts[k])/Rin - (Ts[k]-Tamb[k])/Rout)

All shapes: [N_scenarios, T_timesteps]
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU Batch Simulator] Device: {DEVICE}")


@dataclass
class GPUSimConfig:
    """Physics parameters (same defaults as SimulationConfig in batch_simulator.py)."""
    # ECM (2-RC)
    R0: float = 0.001127
    R1: float = 0.009899
    C1: float = 15290.0
    R2: float = 0.030116
    C2: float = 3236.0
    capacity_ah: float = 3.0       # [Ah]
    ocv_a: float = 3.0             # OCV = a + b*SOC
    ocv_b: float = 1.0
    # EETM (2-state)
    Rin:  float = 3.0              # [K/W]
    Rout: float = 15.0             # [K/W]
    Cc:   float = 30.0             # [J/K]
    Cs:   float = 15.0             # [J/K]
    # Simulation
    dt: float = 1.0                # [s]


def simulate_batch_gpu(
    current_batch:   torch.Tensor,   # [N, T] discharge current [A]
    tamb_batch:      torch.Tensor,   # [N, T] ambient temp [K]
    soc_init_batch:  torch.Tensor,   # [N]    initial SOC
    temp_init_batch: torch.Tensor,   # [N]    initial temperature [K]
    cfg: GPUSimConfig = GPUSimConfig(),
) -> Dict[str, torch.Tensor]:
    """
    Simulate N scenarios simultaneously on GPU.

    Parameters
    ----------
    current_batch   : [N, T] tensor - current profiles (A)
    tamb_batch      : [N, T] tensor - ambient temperature (K)
    soc_init_batch  : [N]    tensor - initial SOC per scenario
    temp_init_batch : [N]    tensor - initial temperature (K) per scenario
    cfg             : GPUSimConfig

    Returns
    -------
    Dict of [N, T] tensors: voltage, soc, temp_core_k, temp_surface_k,
                             temp_ambient_k, heat_generation, power
    """
    N, T = current_batch.shape
    dt = cfg.dt
    C_TO_K = 273.15

    # Pre-compute RC decay factors (scalar)
    alpha1 = float(np.exp(-dt / (cfg.R1 * cfg.C1)))
    alpha2 = float(np.exp(-dt / (cfg.R2 * cfg.C2)))
    Q_nom  = cfg.capacity_ah * 3600.0   # [As]

    # ── Allocate output tensors on GPU ────────────────────────────────────────
    voltage   = torch.zeros(N, T, device=DEVICE, dtype=torch.float32)
    soc_out   = torch.zeros(N, T, device=DEVICE, dtype=torch.float32)
    heat_gen  = torch.zeros(N, T, device=DEVICE, dtype=torch.float32)
    temp_core = torch.zeros(N, T, device=DEVICE, dtype=torch.float32)
    temp_surf = torch.zeros(N, T, device=DEVICE, dtype=torch.float32)

    # ── Initial states [N] ────────────────────────────────────────────────────
    soc  = soc_init_batch.clone().to(DEVICE)          # [N]
    V1   = torch.zeros(N, device=DEVICE)              # RC1 voltage [N]
    V2   = torch.zeros(N, device=DEVICE)              # RC2 voltage [N]
    Tc   = temp_init_batch.clone().to(DEVICE) - C_TO_K  # core temp in °C [N]
    Ts   = Tc.clone()                                 # surface temp in °C [N]

    I_all    = current_batch.to(DEVICE)   # [N, T]
    Tamb_all = tamb_batch.to(DEVICE) - C_TO_K  # [N, T] convert to °C

    # ── Time integration loop (over T steps, each step is N-parallel) ────────
    for k in range(T):
        I    = I_all[:, k]     # [N]  current at step k
        Tamb = Tamb_all[:, k]  # [N]  ambient temp at step k

        # -- ECM --
        OCV = cfg.ocv_a + cfg.ocv_b * soc                         # [N]
        V_term = OCV - I * cfg.R0 - V1 - V2                       # [N]
        Q = I * (OCV - V_term)                                     # [N] heat [W]

        # Store
        soc_out[:, k]  = soc
        voltage[:, k]  = V_term
        heat_gen[:, k] = Q

        # -- EETM --
        temp_core[:, k] = Tc + C_TO_K   # store in K
        temp_surf[:, k] = Ts + C_TO_K

        # -- State update (Euler) --
        soc = soc - I * dt / Q_nom
        soc = soc.clamp(0.0, 1.0)

        V1 = alpha1 * V1 + I * cfg.R1 * (1.0 - alpha1)
        V2 = alpha2 * V2 + I * cfg.R2 * (1.0 - alpha2)

        dTc = (Q - (Tc - Ts) / cfg.Rin) / cfg.Cc
        dTs = ((Tc - Ts) / cfg.Rin - (Ts - Tamb) / cfg.Rout) / cfg.Cs
        Tc  = Tc + dt * dTc
        Ts  = Ts + dt * dTs

    power = voltage * I_all

    return {
        'voltage':       voltage,          # [N, T]
        'soc':           soc_out,          # [N, T]
        'heat_generation': heat_gen,       # [N, T]
        'temp_core_k':   temp_core,        # [N, T]
        'temp_surface_k': temp_surf,       # [N, T]
        'temp_ambient_k': tamb_batch.to(DEVICE),  # [N, T] original K
        'power':         power,            # [N, T]
        'current':       I_all,            # [N, T]
    }


def build_scenario_tensors(
    scenarios:    List[dict],
    drive_loader,
    temp_gen,
    cfg: GPUSimConfig = GPUSimConfig(),
):
    """
    Load all drive cycles & temperature profiles and stack into GPU tensors.
    Pads shorter profiles to match the longest one (with last value).

    Returns
    -------
    current_batch   : [N, T_max]
    tamb_batch      : [N, T_max]
    soc_init_batch  : [N]
    temp_init_batch : [N]
    time_vec        : [T_max]   (time axis in seconds, 1s steps)
    """
    import numpy as np

    N = len(scenarios)
    currents = []
    tambs    = []
    socs     = []
    temps    = []

    for sc in scenarios:
        time, current = drive_loader.load_cycle(sc['drive_cycle'])

        if sc['temp_type'] == 'constant':
            _, tamb_k = temp_gen.generate_constant(
                duration=time[-1], temperature_c=sc['temp_param1'])
        elif sc['temp_type'] == 'sinusoidal':
            _, tamb_k = temp_gen.generate_sinusoidal(
                duration=time[-1],
                temp_mean_c=sc['temp_param1'],
                temp_amplitude_c=sc['temp_param2'],
                period=time[-1])
        else:
            raise ValueError(f"Unknown temp_type: {sc['temp_type']}")

        currents.append(current.astype(np.float32))
        tambs.append(tamb_k.astype(np.float32))
        socs.append(float(sc['soc_initial']))
        temps.append(float(sc.get('temp_initial_k', 298.15)))

    # Pad all profiles to T_max
    T_max = max(len(c) for c in currents)

    def pad(arr):
        if len(arr) < T_max:
            return np.concatenate([arr, np.full(T_max - len(arr), arr[-1])])
        return arr

    current_np = np.stack([pad(c) for c in currents])  # [N, T_max]
    tamb_np    = np.stack([pad(t) for t in tambs])      # [N, T_max]
    soc_np     = np.array(socs,  dtype=np.float32)      # [N]
    temp_np    = np.array(temps, dtype=np.float32)      # [N]
    time_vec   = np.arange(T_max, dtype=np.float32) * cfg.dt

    return (
        torch.from_numpy(current_np),
        torch.from_numpy(tamb_np),
        torch.from_numpy(soc_np),
        torch.from_numpy(temp_np),
        time_vec,
        T_max,
    )


def gpu_simulate_all_scenarios(
    scenarios: List[dict],
    drive_loader,
    temp_gen,
    noise_injector=None,
    add_noise: bool = True,
    cfg: GPUSimConfig = GPUSimConfig(),
    chunk_size: int = 100,          # process in chunks to avoid OOM
) -> List[dict]:
    """
    High-level entry: simulate ALL scenarios on GPU, return list of numpy dicts.

    Parameters
    ----------
    scenarios    : list of scenario dicts (from DatasetBuilder.generate_scenario_list)
    drive_loader : DriveCycleLoader instance
    temp_gen     : TemperatureProfileGenerator instance
    noise_injector: SensorNoiseInjector instance (optional)
    add_noise    : bool
    cfg          : GPUSimConfig
    chunk_size   : how many scenarios to process at once (tune to your GPU VRAM)

    Returns
    -------
    List of result dicts with numpy arrays (per scenario)
    """
    import numpy as np
    C_TO_K = 273.15

    all_results = []
    n_total = len(scenarios)

    for chunk_start in range(0, n_total, chunk_size):
        chunk = scenarios[chunk_start: chunk_start + chunk_size]
        chunk_n = len(chunk)

        print(f"  GPU batch [{chunk_start+1}–{chunk_start+chunk_n}/{n_total}]...", end='\r')

        # Build tensors
        current_t, tamb_t, soc_t, temp_t, time_vec, T_max = build_scenario_tensors(
            chunk, drive_loader, temp_gen, cfg)

        # Run on GPU
        with torch.no_grad():
            gpu_out = simulate_batch_gpu(current_t, tamb_t, soc_t, temp_t, cfg)

        # Download to CPU once for the whole chunk
        cpu = {k: v.cpu().numpy() for k, v in gpu_out.items()}

        for i in range(chunk_n):
            import pandas as pd
            df_dict = {
                'time_s':            time_vec,
                'current_A':         cpu['current'][i],
                'voltage_V':         cpu['voltage'][i],
                'soc':               cpu['soc'][i],
                'temp_surface_C':    cpu['temp_surface_k'][i] - C_TO_K,
                'temp_core_C':       cpu['temp_core_k'][i] - C_TO_K,
                'temp_ambient_C':    cpu['temp_ambient_k'][i] - C_TO_K,
                'heat_generation_W': cpu['heat_generation'][i],
                'power_W':           cpu['power'][i],
            }
            df = pd.DataFrame(df_dict)

            # Add scenario metadata
            sc = chunk[i]
            for key, val in sc.items():
                df[f'meta_{key}'] = val

            # Add sensor noise
            if add_noise and noise_injector is not None:
                clean = {
                    'current':     df['current_A'].values,
                    'voltage':     df['voltage_V'].values,
                    'temp_surface': df['temp_surface_C'].values,
                    'temp_ambient': df['temp_ambient_C'].values,
                }
                noisy = noise_injector.inject_dataset_noise(clean)
                df['current_meas_A']      = noisy['current_noisy']
                df['voltage_meas_V']      = noisy['voltage_noisy']
                df['temp_surface_meas_C'] = noisy['temp_surface_noisy']
                df['temp_ambient_meas_C'] = noisy['temp_ambient_noisy']
                df['power_meas_W']        = noisy['power_noisy']

            all_results.append(df)

    print(f"\n✓ GPU batch simulation complete: {len(all_results)}/{n_total} scenarios")
    return all_results
