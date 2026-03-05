"""
EV Battery Digital Twin — Interactive Streamlit Dashboard
==========================================================
Loads real pipeline outputs (CSVs, trained .pth models) and displays:
  • SOH aging curves per battery  (from Step 2 aging features)
  • ECM voltage validation        (from Step 4 twin dataset)
  • Thermal model validation      (surface temp sim vs real)
  • Core temperature estimation   (physics-twin + transformer)
  • ECM parameter aging evolution  (R0, R1, R2 vs cycle)
  • Training loss curves           (LSTM + Transformer)
  • Multi-ambient & EV drive cycle visualisations
  • Transformer test validation    (predicted vs actual + error)
  • Live inference panel           (LSTM SOH + Transformer core temp)

Run:  streamlit run run_ui_dashboard.py --server.port 8501
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TWIN_CSV = DATA_DIR / "digital_twin_sets" / "augmented_aging_twin_dataset.csv"
EV_CSV = DATA_DIR / "ev_validation_sets" / "ev_drive_cycle_dataset.csv"
PROCESSED = DATA_DIR / "nasa" / "processed"
SOH_MODEL = BASE_DIR / "soh" / "models" / "lstm_residual_soh.pth"
TRANSFORMER_MODEL = BASE_DIR / "transformer" / "models" / "transformer_thermal_core.pth"
NORM_STATS = BASE_DIR / "transformer" / "models" / "normalisation_stats.csv"
PAPER_PLOTS = BASE_DIR / "results" / "paper_plots"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


# ---------------------------------------------------------------------------
# Model definitions (must mirror training code exactly)
# ---------------------------------------------------------------------------
class ResidualLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BatteryThermalTransformer(nn.Module):
    def __init__(self, feature_dim=4, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, src):
        x = self.embedding(src)
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]
        x = self.transformer(x)
        return self.regression_head(x[:, -1, :])


# ---------------------------------------------------------------------------
# Cached data / model loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading digital-twin dataset …")
def load_twin_data():
    if not TWIN_CSV.exists():
        return None
    return pd.read_csv(TWIN_CSV)


@st.cache_data(show_spinner="Loading EV drive-cycle dataset …")
def load_ev_data():
    if not EV_CSV.exists():
        return None
    return pd.read_csv(EV_CSV)


@st.cache_data(show_spinner="Loading aging features …")
def load_aging_features():
    frames = {}
    for b in BATTERIES:
        p = PROCESSED / f"{b}_aging_features.csv"
        if p.exists():
            frames[b] = pd.read_csv(p)
    return frames


@st.cache_resource(show_spinner="Loading LSTM model …")
def load_lstm_model():
    if not SOH_MODEL.exists():
        return None
    model = ResidualLSTM(input_size=3, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(SOH_MODEL, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Transformer model …")
def load_transformer_model():
    if not TRANSFORMER_MODEL.exists():
        return None
    model = BatteryThermalTransformer(
        feature_dim=4, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256, dropout=0.1,
    )
    model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_data(show_spinner="Loading normalisation stats …")
def load_norm_stats():
    if not NORM_STATS.exists():
        return None
    return pd.read_csv(NORM_STATS, index_col=0)


# ---------------------------------------------------------------------------
# Helper: paper-plot images (fallback if plotly fails)
# ---------------------------------------------------------------------------
def show_paper_plot(name, caption):
    p = PAPER_PLOTS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not generated yet. Run `python run_pipeline.py` first (Step 6).")


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------
def page_overview(twin_df, aging_dict):
    """High-level KPIs and aging curves."""
    st.header("📊 Overview — Battery Fleet Health")

    # --- KPI cards ---
    if aging_dict:
        cols = st.columns(len(aging_dict))
        for col, (batt, df) in zip(cols, aging_dict.items()):
            latest_soh = df["soh_true"].iloc[-1]
            n_cycles = len(df)
            r0 = df["r_internal_ohms"].iloc[-1]
            delta_soh = latest_soh - df["soh_true"].iloc[0]
            col.metric(f"{batt} SOH", f"{latest_soh*100:.1f}%",
                       delta=f"{delta_soh*100:+.1f}%")
            col.metric("Cycles", n_cycles)
            col.metric("R₀ (mΩ)", f"{r0*1000:.1f}")
    else:
        st.warning("No aging feature CSVs found. Run Steps 1-2 first.")
        return

    st.markdown("---")

    # --- SOH decay curves ---
    st.subheader("Capacity Fade — All Batteries")
    fig = go.Figure()
    for batt, df in aging_dict.items():
        fig.add_trace(go.Scatter(
            x=df["cycle"], y=df["soh_true"],
            mode="lines+markers", name=batt,
            marker=dict(size=3),
        ))
    fig.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="SOH (State of Health)",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- SOH residual learning ---
    st.subheader("SOH Residual Learning — Physics Baseline vs True")
    sel_batt = st.selectbox("Battery", list(aging_dict.keys()), key="soh_res_batt")
    df = aging_dict[sel_batt]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"{sel_batt}: SOH Components", f"{sel_batt}: Internal Resistance"])
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["soh_true"],
                             mode="lines", name="SOH_true (NASA)", line=dict(color="black", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["soh_physics_baseline"],
                             mode="lines", name="SOH_physics (biased)", line=dict(dash="dash", color="royalblue")), row=1, col=1)
    fig.add_trace(go.Bar(x=df["cycle"], y=df["residual_target"],
                         name="Residual (LSTM target)", marker_color="red", opacity=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["cycle"], y=df["r_internal_ohms"]*1000,
                             mode="lines", name="R_internal (mΩ)", line=dict(color="purple", width=2)), row=1, col=2)
    fig.update_layout(height=400, template="plotly_white")
    fig.update_yaxes(title_text="SOH / Residual", row=1, col=1)
    fig.update_yaxes(title_text="R_internal (mΩ)", row=1, col=2)
    fig.update_xaxes(title_text="Cycle", row=1, col=1)
    fig.update_xaxes(title_text="Cycle", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)


def page_ecm_validation(twin_df):
    """ECM voltage validation plots."""
    st.header("⚡ ECM Voltage Validation (2-RC Model)")
    if twin_df is None:
        st.warning("Digital-twin dataset not found. Run Step 4 first.")
        return

    sel_batt = st.selectbox("Battery", BATTERIES, key="ecm_batt")
    bdf = twin_df[twin_df["battery"] == sel_batt]
    cycles = sorted(bdf["cycle"].unique())
    if not cycles:
        st.warning(f"No cycles for {sel_batt}")
        return

    sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]),
                          int(cycles[len(cycles)//2]), key="ecm_cycle")
    cyc = bdf[bdf["cycle"] == sel_cycle]
    if cyc.empty:
        nearest = min(cycles, key=lambda c: abs(c - sel_cycle))
        cyc = bdf[bdf["cycle"] == nearest]
        sel_cycle = nearest

    soh = cyc["soh_true"].iloc[0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[
                            f"Voltage — {sel_batt} Cycle {sel_cycle} (SOH={soh:.3f})",
                            "Absolute Error"])
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["voltage_V"],
                             mode="lines", name="V_measured", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["voltage_sim_V"],
                             mode="lines", name="V_sim (2-RC ECM)", line=dict(dash="dash", color="orange")), row=1, col=1)
    error_mv = np.abs(cyc["voltage_V"].values - cyc["voltage_sim_V"].values) * 1000
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=error_mv,
                             mode="lines", name="|Error|", line=dict(color="red")), row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="|Error| (mV)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    rmse = np.sqrt(np.mean((cyc["voltage_V"].values - cyc["voltage_sim_V"].values)**2))
    mae = np.mean(np.abs(cyc["voltage_V"].values - cyc["voltage_sim_V"].values))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (mV)", f"{rmse*1000:.2f}")
    c2.metric("MAE (mV)", f"{mae*1000:.2f}")
    c3.metric("Max Error (mV)", f"{error_mv.max():.2f}")


def page_thermal_validation(twin_df):
    """Surface + core temperature validation."""
    st.header("🌡️ Thermal Model Validation (EETM)")
    if twin_df is None:
        st.warning("Digital-twin dataset not found. Run Step 4 first.")
        return

    sel_batt = st.selectbox("Battery", BATTERIES, key="th_batt")
    bdf = twin_df[twin_df["battery"] == sel_batt]
    cycles = sorted(bdf["cycle"].unique())
    if not cycles:
        st.warning(f"No cycles for {sel_batt}")
        return

    sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]),
                          int(cycles[len(cycles)//2]), key="th_cycle")
    cyc = bdf[bdf["cycle"] == sel_cycle]
    if cyc.empty:
        nearest = min(cycles, key=lambda c: abs(c - sel_cycle))
        cyc = bdf[bdf["cycle"] == nearest]
        sel_cycle = nearest

    soh = cyc["soh_true"].iloc[0]

    # Surface temp validation
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[
                            f"Surface Temperature — {sel_batt} Cycle {sel_cycle} (SOH={soh:.3f})",
                            "Core vs Surface Temperature"])
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_C"],
                             mode="lines", name="Ts_measured (NASA)", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_sim_C"],
                             mode="lines", name="Ts_sim (EETM)", line=dict(dash="dash", color="green")), row=1, col=1)

    # Core vs Surface
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_C"],
                             mode="lines", name="T_surface", line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_core_C_TARGET"],
                             mode="lines", name="T_core (physics twin)", line=dict(color="red", width=2)), row=2, col=1)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=650, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    ts_rmse = np.sqrt(np.mean((cyc["temp_surface_C"].values - cyc["temp_surface_sim_C"].values)**2))
    delta_t = cyc["temp_core_C_TARGET"].values - cyc["temp_surface_C"].values
    c1, c2, c3 = st.columns(3)
    c1.metric("Surface RMSE (°C)", f"{ts_rmse:.3f}")
    c2.metric("Max ΔT core−surface (°C)", f"{delta_t.max():.2f}")
    c3.metric("Mean ΔT core−surface (°C)", f"{delta_t.mean():.2f}")


def page_parameter_aging(twin_df):
    """ECM parameter evolution with aging."""
    st.header("🔧 ECM Parameter Aging Evolution")
    if twin_df is None:
        st.warning("Digital-twin dataset not found. Run Step 4 first.")
        return

    cycle_params = twin_df.groupby(["battery", "cycle"]).agg(
        soh=("soh_true", "first"),
        R0=("r0_ohms", "first"),
        R1=("r1_ohms", "first"),
        R2=("r2_ohms", "first"),
    ).reset_index()

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Internal Resistance R₀ Growth", "SOH Capacity Fade"])

    colors = px.colors.qualitative.Set1
    for i, (batt, grp) in enumerate(cycle_params.groupby("battery")):
        fig.add_trace(go.Scatter(
            x=grp["cycle"], y=grp["R0"]*1000,
            mode="lines+markers", name=batt, marker=dict(size=3),
            line=dict(color=colors[i % len(colors)]),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=grp["cycle"], y=grp["soh"],
            mode="lines+markers", name=batt, marker=dict(size=3),
            line=dict(color=colors[i % len(colors)]),
            showlegend=False,
        ), row=1, col=2)

    fig.update_yaxes(title_text="R₀ (mΩ)", row=1, col=1)
    fig.update_yaxes(title_text="SOH", row=1, col=2)
    fig.update_xaxes(title_text="Cycle", row=1, col=1)
    fig.update_xaxes(title_text="Cycle", row=1, col=2)
    fig.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # R1, R2 evolution
    st.subheader("Polarisation Resistances (R₁, R₂)")
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["R₁ Evolution", "R₂ Evolution"])
    for i, (batt, grp) in enumerate(cycle_params.groupby("battery")):
        fig2.add_trace(go.Scatter(
            x=grp["cycle"], y=grp["R1"]*1000,
            mode="lines+markers", name=batt, marker=dict(size=3),
            line=dict(color=colors[i % len(colors)]),
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=grp["cycle"], y=grp["R2"]*1000,
            mode="lines+markers", name=batt, marker=dict(size=3),
            line=dict(color=colors[i % len(colors)]),
            showlegend=False,
        ), row=1, col=2)
    fig2.update_yaxes(title_text="R₁ (mΩ)", row=1, col=1)
    fig2.update_yaxes(title_text="R₂ (mΩ)", row=1, col=2)
    fig2.update_xaxes(title_text="Cycle")
    fig2.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)


def page_live_inference(twin_df, aging_dict, lstm_model, transformer_model, norm_stats):
    """Run live inference on a selected cycle."""
    st.header("🧠 Live AI Inference")

    models_ok = True
    c1, c2 = st.columns(2)
    with c1:
        if lstm_model is not None:
            n_params = sum(p.numel() for p in lstm_model.parameters())
            st.success(f"✅ LSTM Residual SOH loaded ({n_params:,} params)")
        else:
            st.error("❌ LSTM model not found. Run Step 3.")
            models_ok = False
    with c2:
        if transformer_model is not None:
            n_params = sum(p.numel() for p in transformer_model.parameters())
            st.success(f"✅ Transformer Thermal loaded ({n_params:,} params)")
        else:
            st.error("❌ Transformer model not found. Run Step 5.")
            models_ok = False

    st.markdown("---")

    # --- LSTM SOH Inference ---
    st.subheader("📊 SOH Residual Prediction (LSTM)")
    if lstm_model is not None and aging_dict:
        sel_batt = st.selectbox("Battery", list(aging_dict.keys()), key="inf_soh_batt")
        df = aging_dict[sel_batt].copy()
        # Normalise cycle column same as training
        df["cycle_norm"] = df["cycle"] / df["cycle"].max()

        seq_len = 10
        if len(df) > seq_len:
            features = df[["soh_physics_baseline", "r_internal_ohms", "cycle_norm"]].values.astype(np.float32)
            # Predict residual for each possible window
            preds = []
            with torch.no_grad():
                for i in range(len(features) - seq_len):
                    x = torch.from_numpy(features[i:i+seq_len]).unsqueeze(0)
                    pred_res = lstm_model(x).item()
                    preds.append(pred_res)

            pred_cycles = df["cycle"].values[seq_len:]
            physics = df["soh_physics_baseline"].values[seq_len:]
            soh_corrected = physics + np.array(preds)
            soh_true = df["soh_true"].values[seq_len:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_cycles, y=soh_true,
                                     mode="lines", name="SOH_true (NASA)", line=dict(color="black", width=2)))
            fig.add_trace(go.Scatter(x=pred_cycles, y=physics,
                                     mode="lines", name="SOH_physics", line=dict(dash="dash", color="blue")))
            fig.add_trace(go.Scatter(x=pred_cycles, y=soh_corrected,
                                     mode="lines", name="SOH_corrected (physics+LSTM)", line=dict(color="red", width=2)))
            fig.update_layout(
                xaxis_title="Cycle", yaxis_title="SOH",
                title=f"{sel_batt}: LSTM Residual Correction",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            rmse_physics = np.sqrt(np.mean((physics - soh_true)**2))
            rmse_corrected = np.sqrt(np.mean((soh_corrected - soh_true)**2))
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Physics RMSE", f"{rmse_physics:.4f}")
            mc2.metric("Corrected RMSE", f"{rmse_corrected:.4f}")
            improvement = (1 - rmse_corrected / rmse_physics) * 100 if rmse_physics > 0 else 0
            mc3.metric("Improvement", f"{improvement:.1f}%")
    else:
        st.info("Load LSTM model and run Steps 1-3 to enable SOH inference.")

    st.markdown("---")

    # --- Transformer Core Temp Inference ---
    st.subheader("🌡️ Core Temperature Prediction (Transformer)")
    if transformer_model is not None and twin_df is not None and norm_stats is not None:
        sel_batt2 = st.selectbox("Battery", BATTERIES, key="inf_th_batt")
        bdf = twin_df[twin_df["battery"] == sel_batt2]
        cycles = sorted(bdf["cycle"].unique())
        if cycles:
            sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]),
                                  int(cycles[len(cycles)//2]), key="inf_th_cycle")
            cyc = bdf[bdf["cycle"] == sel_cycle]
            if cyc.empty:
                nearest = min(cycles, key=lambda c: abs(c - sel_cycle))
                cyc = bdf[bdf["cycle"] == nearest]

            # Normalise inputs using saved stats
            feat_cols = ["current_A", "voltage_V", "r0_ohms", "temp_surface_C"]
            target_col = "temp_core_C_TARGET"

            data_norm = cyc[feat_cols].copy()
            for col in feat_cols:
                mu = norm_stats.loc["mean", col]
                sigma = norm_stats.loc["std", col]
                data_norm[col] = (data_norm[col] - mu) / sigma

            target_mu = norm_stats.loc["mean", target_col]
            target_sigma = norm_stats.loc["std", target_col]

            # Determine window size (match training logic)
            avg_pts = twin_df.groupby(["battery", "cycle"]).size().mean()
            window_size = min(60, int(avg_pts * 0.3))
            window_size = max(10, window_size)

            if len(data_norm) > window_size:
                preds_core = []
                data_arr = data_norm[feat_cols].values.astype(np.float32)
                with torch.no_grad():
                    for i in range(len(data_arr) - window_size):
                        x = torch.from_numpy(data_arr[i:i+window_size]).unsqueeze(0)
                        pred_norm = transformer_model(x).item()
                        pred_real = pred_norm * target_sigma + target_mu
                        preds_core.append(pred_real)

                time_pred = cyc["time_s"].values[window_size:]
                core_true = cyc[target_col].values[window_size:]
                surface = cyc["temp_surface_C"].values[window_size:]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_pred, y=surface,
                                         mode="lines", name="T_surface (measured)", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=time_pred, y=core_true,
                                         mode="lines", name="T_core (physics twin)", line=dict(color="red", width=2)))
                fig.add_trace(go.Scatter(x=time_pred, y=preds_core,
                                         mode="lines", name="T_core (Transformer pred)", line=dict(color="green", dash="dash", width=2)))
                fig.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Temperature (°C)",
                    title=f"Core Temperature Prediction — {sel_batt2} Cycle {sel_cycle}",
                    template="plotly_white", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                rmse_core = np.sqrt(np.mean((np.array(preds_core) - core_true)**2))
                mae_core = np.mean(np.abs(np.array(preds_core) - core_true))
                mc1, mc2 = st.columns(2)
                mc1.metric("Core Temp RMSE (°C)", f"{rmse_core:.4f}")
                mc2.metric("Core Temp MAE (°C)", f"{mae_core:.4f}")
            else:
                st.warning("Not enough data points in this cycle for the window size.")
    else:
        st.info("Load Transformer model and run Steps 1-5 to enable core temp inference.")


def page_paper_plots():
    """Show pre-generated paper-quality plots from Step 6."""
    st.header("📄 Paper-Quality Figures (Step 6)")
    st.markdown("These are the matplotlib figures generated by `reports/generate_paper_plots.py`.")

    plots = [
        ("fig1_voltage_validation.png", "Fig 1: ECM Voltage Validation"),
        ("fig2_surface_temp_validation.png", "Fig 2: EETM Surface Temperature Validation"),
        ("fig3_core_temperature.png", "Fig 3: Core vs Surface Temperature"),
        ("fig4_parameter_aging.png", "Fig 4: ECM Parameter Aging Evolution"),
        ("fig5_soh_residual.png", "Fig 5: SOH Residual Learning"),
        ("fig6_drive_thermal.png", "Fig 6: Drive Current & Thermal Response"),
        ("transformer_test_validation.png", "Fig 7: Transformer Test Validation (Predicted vs Actual + Error)"),
        ("ev_us06_transformer_validation.png", "Fig 8: EV US06 Drive-Cycle Transformer Validation"),
    ]
    found = 0
    for fname, caption in plots:
        p = PAPER_PLOTS / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
            found += 1
            st.markdown("---")
    if found == 0:
        st.warning("No paper plots found. Run `python run_pipeline.py` (Step 6) to generate them.")


def page_training_curves():
    """Show training and validation loss curves for LSTM and Transformer."""
    st.header("📈 Training & Validation Loss Curves")
    st.markdown("These loss curves demonstrate model convergence over 100 epochs with 80/20 train/val split.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LSTM Residual SOH (Step 3)")
        lstm_plot = PAPER_PLOTS / "lstm_training_loss.png"
        if lstm_plot.exists():
            st.image(str(lstm_plot), caption="LSTM Train/Val Loss over 100 Epochs",
                     use_container_width=True)
        else:
            st.info("LSTM loss curve not yet generated. Run Step 3.")

    with col2:
        st.subheader("Transformer Core Temp (Step 5)")
        tf_plot = PAPER_PLOTS / "transformer_training_loss.png"
        if tf_plot.exists():
            st.image(str(tf_plot), caption="Transformer Train/Val Loss over 100 Epochs",
                     use_container_width=True)
        else:
            st.info("Transformer loss curve not yet generated. Run Step 5.")

    st.markdown("---")
    st.markdown("""
    **Training Details:**
    - **LSTM:** input_size=3, hidden=64, 1 layer, Adam (lr=0.005), batch=16, 100 epochs
    - **Transformer:** d_model=128, nhead=4, 4 encoder layers, dim_ff=256, dropout=0.1, Adam (lr=5e-4), 100 epochs
    - **Data split:** 80% train / 20% validation via `sklearn.model_selection.train_test_split` (random_state=42)
    """)


def page_multi_ambient_ev():
    """Show multi-ambient drive cycle visualisations and EV dataset explorer."""
    st.header("🚗 Multi-Ambient Drive Cycles & EV Validation Data")

    # --- Multi-ambient visualisation plots (from Step 4) ---
    st.subheader("Aggressive & Mixed Drive Profiles at Multiple Temperatures")
    st.markdown("Physics-based simulation of synthetic drive profiles at 0°C, 20°C, and 50°C ambient.")

    c1, c2 = st.columns(2)
    with c1:
        agg_plot = PAPER_PLOTS / "aggressive_multi_temp_visualization.png"
        if agg_plot.exists():
            st.image(str(agg_plot), caption="Aggressive Profile: Multi-Ambient Thermal",
                     use_container_width=True)
        else:
            st.info("Aggressive multi-temp plot not generated yet.")
    with c2:
        mix_plot = PAPER_PLOTS / "mixed_multi_temp_visualization.png"
        if mix_plot.exists():
            st.image(str(mix_plot), caption="Mixed Profile: Multi-Ambient Thermal",
                     use_container_width=True)
        else:
            st.info("Mixed multi-temp plot not generated yet.")

    # --- US06 Triple Stack ---
    st.subheader("US06 EV Drive Cycle — Triple Stack")
    us06_plot = PAPER_PLOTS / "us06_ev_triple_stack.png"
    if us06_plot.exists():
        st.image(str(us06_plot), caption="US06 at 0°C, 25°C, 45°C — Current, Core Temp, Voltage",
                 use_container_width=True)
    else:
        st.info("US06 triple-stack plot not generated yet.")

    st.markdown("---")

    # --- EV Dataset Explorer ---
    st.subheader("EV Drive-Cycle Dataset Explorer (288 Simulations)")
    ev_df = load_ev_data()
    if ev_df is None:
        st.warning("EV drive-cycle dataset not found. Run Step 4 first.")
        return

    st.info(f"**Dataset size:** {len(ev_df):,} rows  •  "
            f"**Unique simulations:** {ev_df['battery'].nunique()}")

    # Parse battery names for filtering
    batt_names = sorted(ev_df["battery"].unique())

    # Extract drive cycle type from battery name (e.g., "B0005_cyc20_UDDS_T0")
    drive_types = sorted(set(n.split("_")[2] for n in batt_names if len(n.split("_")) >= 3))
    temp_types = sorted(set(n.split("_")[3] for n in batt_names if len(n.split("_")) >= 4))

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_drive = st.selectbox("Drive Cycle", drive_types, key="ev_drive")
    with c2:
        sel_temp = st.selectbox("Temperature", temp_types, key="ev_temp")
    with c3:
        filtered_batts = [b for b in batt_names if sel_drive in b and sel_temp in b]
        sel_sim = st.selectbox("Simulation", filtered_batts[:20], key="ev_sim")

    if sel_sim:
        sim_df = ev_df[ev_df["battery"] == sel_sim]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=[
                                f"Current Profile ({sel_sim})",
                                "Voltage Response",
                                "Temperature (Surface vs Core)"])

        fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["current_A"],
                                 mode="lines", name="Current (A)", line=dict(color="black")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["voltage_V"],
                                 mode="lines", name="Voltage (V)", line=dict(color="blue")),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["temp_surface_C"],
                                 mode="lines", name="T_surface", line=dict(color="blue")),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["temp_core_C_TARGET"],
                                 mode="lines", name="T_core", line=dict(color="red", width=2)),
                      row=3, col=1)

        fig.update_yaxes(title_text="Current (A)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
        fig.update_yaxes(title_text="Temp (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Duration (s)", f"{sim_df['time_s'].max():.0f}")
        mc2.metric("SOH", f"{sim_df['soh_true'].iloc[0]:.3f}")
        delta_tc = sim_df["temp_core_C_TARGET"].values - sim_df["temp_surface_C"].values
        mc3.metric("Max ΔT core−surface (°C)", f"{delta_tc.max():.2f}")
        mc4.metric("Data Points", f"{len(sim_df):,}")


def page_transformer_validation(twin_df):
    """Interactive Transformer test validation: predicted vs actual + error."""
    st.header("🎯 Transformer Test Validation")

    if twin_df is None:
        st.warning("Digital-twin dataset not found. Run Step 4 first.")
        return

    # Show pre-generated matplotlib plots first
    st.subheader("Pre-generated Validation Plots (from Step 6)")
    val_plot = PAPER_PLOTS / "transformer_test_validation.png"
    if val_plot.exists():
        st.image(str(val_plot), caption="Transformer Test Validation — Predicted vs Actual Tc + Error",
                 use_container_width=True)
    else:
        st.info("Transformer validation plot not yet generated. Run Step 6.")

    ev_val_plot = PAPER_PLOTS / "ev_us06_transformer_validation.png"
    if ev_val_plot.exists():
        st.image(str(ev_val_plot), caption="EV US06 Transformer Validation",
                 use_container_width=True)

    st.markdown("---")

    # Interactive validation
    st.subheader("Interactive Validation — Pick Any Cycle")
    transformer_model = load_transformer_model()
    norm_stats = load_norm_stats()

    if transformer_model is None or norm_stats is None:
        st.warning("Transformer model or normalisation stats not loaded. Run Steps 4-5.")
        return

    sel_batt = st.selectbox("Battery", BATTERIES, key="tv_batt")
    bdf = twin_df[twin_df["battery"] == sel_batt]
    cycles = sorted(bdf["cycle"].unique())
    if not cycles:
        st.warning(f"No cycles for {sel_batt}")
        return

    sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]),
                          int(cycles[-3] if len(cycles) > 3 else cycles[-1]), key="tv_cycle")
    cyc = bdf[bdf["cycle"] == sel_cycle]
    if cyc.empty:
        nearest = min(cycles, key=lambda c: abs(c - sel_cycle))
        cyc = bdf[bdf["cycle"] == nearest]
        sel_cycle = nearest

    feat_cols = ["current_A", "voltage_V", "r0_ohms", "temp_surface_C"]
    target_col = "temp_core_C_TARGET"

    data_norm = cyc[feat_cols].copy()
    for col in feat_cols:
        mu = norm_stats.loc["mean", col]
        sigma = norm_stats.loc["std", col]
        data_norm[col] = (data_norm[col] - mu) / sigma

    target_mu = norm_stats.loc["mean", target_col]
    target_sigma = norm_stats.loc["std", target_col]

    avg_pts = twin_df.groupby(["battery", "cycle"]).size().mean()
    window_size = min(60, int(avg_pts * 0.3))
    window_size = max(10, window_size)

    if len(data_norm) <= window_size:
        st.warning("Not enough data points in this cycle for the window size.")
        return

    data_arr = data_norm[feat_cols].values.astype(np.float32)
    preds_core = []
    with torch.no_grad():
        for i in range(len(data_arr) - window_size):
            x = torch.from_numpy(data_arr[i:i+window_size]).unsqueeze(0)
            pred_norm = transformer_model(x).item()
            pred_real = pred_norm * target_sigma + target_mu
            preds_core.append(pred_real)

    time_pred = cyc["time_s"].values[window_size:]
    core_true = cyc[target_col].values[window_size:]
    preds_arr = np.array(preds_core)
    error = preds_arr - core_true

    # Top: prediction comparison
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        subplot_titles=[
                            f"Core Temperature — {sel_batt} Cycle {sel_cycle} (SOH={cyc['soh_true'].iloc[0]:.3f})",
                            "Estimation Error"])
    fig.add_trace(go.Scatter(x=time_pred, y=core_true,
                             mode="lines", name="Tc Actual (Physics Twin)",
                             line=dict(color="blue", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_pred, y=preds_arr,
                             mode="lines", name="Tc Predicted (Transformer)",
                             line=dict(color="orange", width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_pred, y=error,
                             mode="lines", name="Error",
                             line=dict(color="red", width=1)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Error (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("RMSE (°C)", f"{rmse:.4f}")
    mc2.metric("MAE (°C)", f"{mae:.4f}")
    mc3.metric("Max |Error| (°C)", f"{np.abs(error).max():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="EV Battery Digital Twin",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔋 EV Battery Digital Twin Dashboard")
    st.caption(
        "Hybrid Electrical Circuit Model & Deep Learning-Based Core Temperature Estimation  •  "
        "Based on Samanta, Surya, Williamson et al. (IEEE TTE 2022)"
    )

    # --- Sidebar ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Page", [
        "📊 Overview",
        "⚡ ECM Voltage Validation",
        "🌡️ Thermal Validation",
        "🔧 Parameter Aging",
        "📈 Training Loss Curves",
        "🚗 EV Drive Cycles",
        "🎯 Transformer Validation",
        "🧠 Live Inference",
        "📄 Paper Plots",
    ])

    st.sidebar.markdown("---")
    st.sidebar.header("Pipeline Status")

    # Check data availability
    twin_df = load_twin_data()
    ev_df = load_ev_data()
    aging_dict = load_aging_features()
    lstm_model = load_lstm_model()
    transformer_model = load_transformer_model()
    norm_stats = load_norm_stats()

    # Status indicators
    checks = {
        "NASA .mat files": any((DATA_DIR / "nasa" / f"{b}.mat").exists() for b in BATTERIES),
        "Processed CSVs (Step 2)": bool(aging_dict),
        "Digital Twin CSV (Step 4)": twin_df is not None,
        "EV Drive-Cycle CSV (Step 4)": ev_df is not None,
        "LSTM Model (Step 3)": lstm_model is not None,
        "Transformer Model (Step 5)": transformer_model is not None,
        "Normalisation Stats": norm_stats is not None,
        "Training Loss Curves": (PAPER_PLOTS / "lstm_training_loss.png").exists(),
        "Paper Plots (Step 6)": PAPER_PLOTS.exists() and any(PAPER_PLOTS.glob("fig*.png")),
    }
    for label, ok in checks.items():
        if ok:
            st.sidebar.success(f"✅ {label}")
        else:
            st.sidebar.warning(f"⚠️ {label}")

    st.sidebar.markdown("---")
    dataset_info_parts = []
    if twin_df is not None:
        dataset_info_parts.append(
            f"**NASA Twin:** {len(twin_df):,} rows, "
            f"{twin_df['battery'].nunique()} batteries, "
            f"{twin_df.groupby('battery')['cycle'].nunique().sum()} cycles"
        )
    if ev_df is not None:
        dataset_info_parts.append(
            f"**EV Drive-Cycle:** {len(ev_df):,} rows, "
            f"{ev_df['battery'].nunique()} simulations"
        )
    if dataset_info_parts:
        st.sidebar.info("\n\n".join(dataset_info_parts))

    # --- Page routing ---
    if page == "📊 Overview":
        page_overview(twin_df, aging_dict)
    elif page == "⚡ ECM Voltage Validation":
        page_ecm_validation(twin_df)
    elif page == "🌡️ Thermal Validation":
        page_thermal_validation(twin_df)
    elif page == "🔧 Parameter Aging":
        page_parameter_aging(twin_df)
    elif page == "📈 Training Loss Curves":
        page_training_curves()
    elif page == "🚗 EV Drive Cycles":
        page_multi_ambient_ev()
    elif page == "🎯 Transformer Validation":
        page_transformer_validation(twin_df)
    elif page == "🧠 Live Inference":
        page_live_inference(twin_df, aging_dict, lstm_model, transformer_model, norm_stats)
    elif page == "📄 Paper Plots":
        page_paper_plots()


if __name__ == "__main__":
    main()
