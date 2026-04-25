"""
EV Battery Digital Twin — Interactive Streamlit Dashboard
==========================================================
Loads real pipeline outputs (CSVs, trained .pth models) and displays:
  • SOH aging curves per battery  (from Step 2 aging features)
  • ECM voltage validation        (from Step 4 twin dataset)
  • Thermal model validation      (surface temp sim vs real)
  • Core temperature estimation   (physics-twin + transformer with uncertainty)
  • ECM parameter aging evolution  (R0, R1, R2 vs cycle)
  • Training loss curves           (LSTM + Transformer)
  • Multi-ambient & EV drive cycle visualisations
  • Transformer test validation    (predicted vs actual + error + 95% CI)
  • Live inference panel           (LSTM SOH + Transformer core temp with CI)
"""

import os, sys, streamlit as st, pandas as pd, numpy as np, torch, torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths
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
# Model definitions (with MC dropout uncertainty)
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
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = dropout

    def forward(self, src, mc_dropout=False):
        if mc_dropout:
            self.train()  # enables dropout for MC sampling
        x = self.embedding(src)
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]
        x = self.transformer(x)
        return self.regression_head(x[:, -1, :])

    def predict_with_uncertainty(self, x, n_samples=50):
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x, mc_dropout=True).cpu().numpy())
        preds = np.array(preds).squeeze(-1)   # (n_samples, batch)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading digital-twin dataset …")
def load_twin_data():
    if not TWIN_CSV.exists(): return None
    return pd.read_csv(TWIN_CSV)

@st.cache_data(show_spinner="Loading EV drive-cycle dataset …")
def load_ev_data():
    if not EV_CSV.exists(): return None
    return pd.read_csv(EV_CSV)

@st.cache_data(show_spinner="Loading aging features …")
def load_aging_features():
    frames = {}
    for b in BATTERIES:
        p = PROCESSED / f"{b}_aging_features.csv"
        if p.exists(): frames[b] = pd.read_csv(p)
    return frames

@st.cache_resource(show_spinner="Loading LSTM model …")
def load_lstm_model():
    if not SOH_MODEL.exists(): return None
    model = ResidualLSTM(3, 64, 1)
    model.load_state_dict(torch.load(SOH_MODEL, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_resource(show_spinner="Loading Transformer model …")
def load_transformer_model():
    if not TRANSFORMER_MODEL.exists(): return None
    model = BatteryThermalTransformer(feature_dim=4, d_model=128, nhead=4,
                                      num_layers=4, dim_feedforward=256, dropout=0.1)
    model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_data(show_spinner="Loading normalisation stats …")
def load_norm_stats():
    if not NORM_STATS.exists(): return None
    return pd.read_csv(NORM_STATS, index_col=0)

# ---------------------------------------------------------------------------
# Helper: interpolate a single cycle to 1‑second grid
# ---------------------------------------------------------------------------
def interpolate_cycle(df_cycle):
    df = df_cycle.sort_values('time_s')
    if len(df) < 2:
        return df
    t_old = df['time_s'].values
    t_new = np.arange(t_old[0], t_old[-1] + 0.5, 1.0)
    new_data = {'time_s': t_new}
    for col in df.columns:
        if col != 'time_s':
            f = interp1d(t_old, df[col].values, kind='linear', fill_value='extrapolate')
            new_data[col] = f(t_new)
    return pd.DataFrame(new_data)

# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------
def page_overview(twin_df, aging_dict):
    st.header("📊 Overview — Battery Fleet Health")
    if aging_dict:
        cols = st.columns(len(aging_dict))
        for col, (batt, df) in zip(cols, aging_dict.items()):
            latest_soh = df["soh_true"].iloc[-1]
            n_cycles = len(df)
            r0 = df["r_internal_ohms"].iloc[-1]
            delta_soh = latest_soh - df["soh_true"].iloc[0]
            col.metric(f"{batt} SOH", f"{latest_soh*100:.1f}%", delta=f"{delta_soh*100:+.1f}%")
            col.metric("Cycles", n_cycles)
            col.metric("R₀ (mΩ)", f"{r0*1000:.1f}")
    else:
        st.warning("No aging feature CSVs found.")
        return
    st.markdown("---")
    st.subheader("Capacity Fade — All Batteries")
    fig = go.Figure()
    for batt, df in aging_dict.items():
        fig.add_trace(go.Scatter(x=df["cycle"], y=df["soh_true"], mode="lines+markers", name=batt))
    fig.update_layout(xaxis_title="Discharge Cycle", yaxis_title="SOH", template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

def page_ecm_validation(twin_df):
    st.header("⚡ ECM Voltage Validation (2-RC Model)")
    if twin_df is None: st.warning("Digital-twin dataset not found."); return
    sel_batt = st.selectbox("Battery", BATTERIES, key="ecm_batt")
    bdf = twin_df[twin_df["battery"] == sel_batt]
    cycles = sorted(bdf["cycle"].unique())
    if not cycles: st.warning("No cycles"); return
    sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]), int(cycles[len(cycles)//2]), key="ecm_cycle")
    cyc = bdf[bdf["cycle"] == sel_cycle]
    if cyc.empty: nearest = min(cycles, key=lambda c: abs(c - sel_cycle)); cyc = bdf[bdf["cycle"] == nearest]
    soh = cyc["soh_true"].iloc[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["voltage_V"], name="V_measured", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["voltage_sim_V"], name="V_sim (2-RC ECM)", line=dict(dash="dash", color="orange")), row=1, col=1)
    error_mv = np.abs(cyc["voltage_V"].values - cyc["voltage_sim_V"].values) * 1000
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=error_mv, name="|Error|", line=dict(color="red")), row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1); fig.update_yaxes(title_text="|Error| (mV)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    rmse = np.sqrt(np.mean((cyc["voltage_V"].values - cyc["voltage_sim_V"].values)**2))
    mae = np.mean(np.abs(cyc["voltage_V"].values - cyc["voltage_sim_V"].values))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (mV)", f"{rmse*1000:.2f}"); c2.metric("MAE (mV)", f"{mae*1000:.2f}"); c3.metric("Max Error (mV)", f"{error_mv.max():.2f}")

def page_thermal_validation(twin_df):
    st.header("🌡️ Thermal Model Validation (EETM)")
    if twin_df is None: st.warning("No twin data"); return
    sel_batt = st.selectbox("Battery", BATTERIES, key="th_batt")
    bdf = twin_df[twin_df["battery"] == sel_batt]
    cycles = sorted(bdf["cycle"].unique())
    sel_cycle = st.slider("Cycle", int(cycles[0]), int(cycles[-1]), int(cycles[len(cycles)//2]), key="th_cycle")
    cyc = bdf[bdf["cycle"] == sel_cycle]
    if cyc.empty: nearest = min(cycles, key=lambda c: abs(c - sel_cycle)); cyc = bdf[bdf["cycle"] == nearest]
    soh = cyc["soh_true"].iloc[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_C"], name="Ts_measured", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_sim_C"], name="Ts_sim (EETM)", line=dict(dash="dash", color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_surface_C"], name="T_surface", line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=cyc["time_s"], y=cyc["temp_core_C_TARGET"], name="T_core (physics twin)", line=dict(color="red", width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1); fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)
    ts_rmse = np.sqrt(np.mean((cyc["temp_surface_C"].values - cyc["temp_surface_sim_C"].values)**2))
    delta_t = cyc["temp_core_C_TARGET"].values - cyc["temp_surface_C"].values
    c1, c2, c3 = st.columns(3)
    c1.metric("Surface RMSE (°C)", f"{ts_rmse:.3f}")
    c2.metric("Max ΔT core−surface (°C)", f"{delta_t.max():.2f}")
    c3.metric("Mean ΔT core−surface (°C)", f"{delta_t.mean():.2f}")

def page_parameter_aging(twin_df):
    st.header("🔧 ECM Parameter Aging Evolution")
    if twin_df is None: return
    cycle_params = twin_df.groupby(["battery", "cycle"]).agg(
        soh=("soh_true", "first"), R0=("r0_ohms", "first"),
        R1=("r1_ohms", "first"), R2=("r2_ohms", "first")).reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["R₀ Growth", "SOH Capacity Fade"])
    colors = px.colors.qualitative.Set1
    for i, (batt, grp) in enumerate(cycle_params.groupby("battery")):
        fig.add_trace(go.Scatter(x=grp["cycle"], y=grp["R0"]*1000, name=batt, marker=dict(size=3), line=dict(color=colors[i%len(colors)])), row=1, col=1)
        fig.add_trace(go.Scatter(x=grp["cycle"], y=grp["soh"], name=batt, showlegend=False, marker=dict(size=3), line=dict(color=colors[i%len(colors)])), row=1, col=2)
    fig.update_yaxes(title_text="R₀ (mΩ)", row=1, col=1); fig.update_yaxes(title_text="SOH", row=1, col=2)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

def page_live_inference(twin_df, aging_dict, lstm_model, transformer_model, norm_stats):
    st.header("🧠 Live AI Inference")
    c1, c2 = st.columns(2)
    with c1:
        if lstm_model: st.success("✅ LSTM loaded")
        else: st.error("❌ LSTM not found")
    with c2:
        if transformer_model: st.success("✅ Transformer loaded")
        else: st.error("❌ Transformer not found")
    st.markdown("---")

    # LSTM part unchanged (10‑cycle sliding window)
    st.subheader("📊 SOH Residual Prediction (LSTM)")
    if lstm_model and aging_dict:
        sel_batt = st.selectbox("Battery", list(aging_dict.keys()), key="inf_soh_batt")
        df = aging_dict[sel_batt].copy()
        df["cycle_norm"] = df["cycle"] / df["cycle"].max()
        seq_len = 10
        if len(df) > seq_len:
            features = df[["soh_physics_baseline", "r_internal_ohms", "cycle_norm"]].values.astype(np.float32)
            preds = []
            with torch.no_grad():
                for i in range(len(features)-seq_len):
                    x = torch.from_numpy(features[i:i+seq_len]).unsqueeze(0)
                    preds.append(lstm_model(x).item())
            pred_cycles = df["cycle"].values[seq_len:]
            physics = df["soh_physics_baseline"].values[seq_len:]
            soh_corrected = physics + np.array(preds)
            soh_true = df["soh_true"].values[seq_len:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_cycles, y=soh_true, name="SOH_true"))
            fig.add_trace(go.Scatter(x=pred_cycles, y=physics, name="SOH_physics", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=pred_cycles, y=soh_corrected, name="SOH_corrected"))
            st.plotly_chart(fig, use_container_width=True)

    # Transformer inference with interpolation and MC dropout
    st.subheader("🌡️ Core Temperature Prediction (Transformer)")
    if transformer_model and twin_df is not None and norm_stats is not None:
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
            # Interpolate to 1 s for consistent window size
            cyc = interpolate_cycle(cyc)
            feat_cols = ["current_A", "voltage_V", "r0_ohms", "temp_surface_C"]
            target_col = "temp_core_C_TARGET"
            data_norm = cyc[feat_cols].copy()
            for col in feat_cols:
                mu = norm_stats.loc["mean", col]; sigma = norm_stats.loc["std", col]
                data_norm[col] = (data_norm[col] - mu) / sigma
            target_mu = norm_stats.loc["mean", target_col]; target_sigma = norm_stats.loc["std", target_col]
            window_size = 60   # must match training
            if len(data_norm) > window_size:
                mean_preds, std_preds = [], []
                data_arr = data_norm[feat_cols].values.astype(np.float32)
                with torch.no_grad():
                    for i in range(len(data_arr) - window_size):
                        x = torch.from_numpy(data_arr[i:i+window_size]).unsqueeze(0)
                        m, s = transformer_model.predict_with_uncertainty(x, n_samples=50)
                        mean_preds.append(m[0]); std_preds.append(s[0])
                time_pred = cyc["time_s"].values[window_size:]
                mean_arr = np.array(mean_preds) * target_sigma + target_mu
                std_arr  = np.array(std_preds) * target_sigma
                core_true = cyc[target_col].values[window_size:]
                surface = cyc["temp_surface_C"].values[window_size:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_pred, y=surface, name="T_surface (measured)", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=time_pred, y=core_true, name="T_core (physics twin)", line=dict(color="red", width=2)))
                fig.add_trace(go.Scatter(x=time_pred, y=mean_arr, name="T_core (Transformer)", line=dict(color="green", dash="dash")))
                upper = mean_arr + 2*std_arr; lower = mean_arr - 2*std_arr
                fig.add_trace(go.Scatter(x=time_pred, y=upper, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=time_pred, y=lower, fill='tonexty', mode='lines', line=dict(width=0), name='95% CI'))
                st.plotly_chart(fig, use_container_width=True)
                rmse = np.sqrt(np.mean((mean_arr - core_true)**2))
                st.metric("RMSE (°C)", f"{rmse:.4f}")
            else:
                st.warning("Not enough data points in this cycle for the window size.")

def page_paper_plots():
    st.header("📄 Paper-Quality Figures")
    plots = [("fig1_voltage_validation.png","Fig1"),("fig2_surface_temp_validation.png","Fig2"),
             ("fig3_core_temperature.png","Fig3"),("fig4_parameter_aging.png","Fig4"),
             ("fig5_soh_residual.png","Fig5"),("fig6_drive_thermal.png","Fig6"),
             ("transformer_test_validation.png","Fig7"),("ev_us06_transformer_validation.png","Fig8")]
    for fname, cap in plots:
        p = PAPER_PLOTS / fname
        if p.exists(): st.image(str(p), caption=cap, use_container_width=True)

def page_training_curves():
    st.header("📈 Training Loss Curves")
    col1, col2 = st.columns(2)
    with col1:
        lstm_plot = PAPER_PLOTS / "lstm_training_loss.png"
        if lstm_plot.exists(): st.image(str(lstm_plot))
    with col2:
        tf_plot = PAPER_PLOTS / "transformer_training_loss.png"
        if tf_plot.exists(): st.image(str(tf_plot))

def page_multi_ambient_ev():
    st.header("🚗 Multi-Ambient Drive Cycles & EV Validation Data")
    st.image(str(PAPER_PLOTS/"aggressive_multi_temp_visualization.png"), caption="Aggressive")
    st.image(str(PAPER_PLOTS/"mixed_multi_temp_visualization.png"), caption="Mixed")
    ev_df = load_ev_data()
    if ev_df is not None:
        st.subheader("EV Dataset Explorer")
        batt_names = sorted(ev_df["battery"].unique())
        drive_types = sorted(set(n.split("_")[2] for n in batt_names if len(n.split("_"))>=3))
        temp_types = sorted(set(n.split("_")[3] for n in batt_names if len(n.split("_"))>=4))
        c1, c2, c3 = st.columns(3)
        sel_drive = c1.selectbox("Drive Cycle", drive_types, key="ev_drive")
        sel_temp = c2.selectbox("Temperature", temp_types, key="ev_temp")
        filtered_batts = [b for b in batt_names if sel_drive in b and sel_temp in b]
        sel_sim = c3.selectbox("Simulation", filtered_batts[:20])
        if sel_sim:
            sim_df = ev_df[ev_df["battery"]==sel_sim]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
            fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["current_A"], name="Current"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["voltage_V"], name="Voltage"), row=2, col=1)
            fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["temp_surface_C"], name="T_surface"), row=3, col=1)
            fig.add_trace(go.Scatter(x=sim_df["time_s"], y=sim_df["temp_core_C_TARGET"], name="T_core"), row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

def page_transformer_validation(twin_df):
    st.header("🎯 Transformer Test Validation")
    st.image(str(PAPER_PLOTS/"transformer_test_validation.png"), caption="Transformer Validation")
    st.image(str(PAPER_PLOTS/"ev_us06_transformer_validation.png"), caption="EV US06 Validation")

def main():
    st.set_page_config(page_title="EV Battery Digital Twin", layout="wide")
    st.title("🔋 EV Battery Digital Twin Dashboard")
    page = st.sidebar.radio("Page", ["📊 Overview","⚡ ECM Voltage Validation","🌡️ Thermal Validation",
                                     "🔧 Parameter Aging","📈 Training Loss Curves","🚗 EV Drive Cycles",
                                     "🎯 Transformer Validation","🧠 Live Inference","📄 Paper Plots"])
    twin_df = load_twin_data(); aging_dict = load_aging_features()
    lstm_model = load_lstm_model(); transformer_model = load_transformer_model(); norm_stats = load_norm_stats()
    if page == "📊 Overview": page_overview(twin_df, aging_dict)
    elif page == "⚡ ECM Voltage Validation": page_ecm_validation(twin_df)
    elif page == "🌡️ Thermal Validation": page_thermal_validation(twin_df)
    elif page == "🔧 Parameter Aging": page_parameter_aging(twin_df)
    elif page == "📈 Training Loss Curves": page_training_curves()
    elif page == "🚗 EV Drive Cycles": page_multi_ambient_ev()
    elif page == "🎯 Transformer Validation": page_transformer_validation(twin_df)
    elif page == "🧠 Live Inference": page_live_inference(twin_df, aging_dict, lstm_model, transformer_model, norm_stats)
    elif page == "📄 Paper Plots": page_paper_plots()

if __name__ == "__main__":
    main()