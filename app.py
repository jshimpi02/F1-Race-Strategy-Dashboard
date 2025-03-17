import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
import pygad
import time


# === STREAMLIT CONFIG ===
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# === DARK/LIGHT MODE ===
mode = st.sidebar.radio("Select Mode", ("Dark", "Light"))
if mode == "Dark":
    st.markdown("""
        <style>
            body { background-color: #0E1117; color: #FAFAFA; }
            .css-1aumxhk { background-color: #0E1117; }
        </style>
        """, unsafe_allow_html=True)

# === HEADER ===
st.title("🏎️ F1 Race Strategy Simulator - Silverstone Circuit Edition")
st.sidebar.image("assets/f1_logo.png", width=150)

# === TEAM & DRIVER CONFIG ===
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]}
}

selected_team = st.sidebar.selectbox("Select Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_photo_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
team_colors = teams[selected_team]["color"]

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_photo_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS ===
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 52)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 18, 25, 22)

# === WEATHER SETTINGS ===
weather = st.sidebar.radio("Weather Condition", ("Clear", "Light Rain", "Heavy Rain"))

# === RL ENVIRONMENT ===
class F1PitStopEnv(gym.Env):
    def __init__(self):
        super(F1PitStopEnv, self).__init__()
        self.action_space = spaces.Discrete(race_length)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 0
        self.total_time = 0
        obs = np.array([0, 0, 0], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        pit = 1 if action == self.current_lap else 0
        lap_time = 90 + (0.20 * self.current_lap) + pit * pit_stop_time
        self.total_time += lap_time
        reward = -lap_time
        self.current_lap += 1
        done = self.current_lap >= race_length
        obs = np.array([self.current_lap / race_length, lap_time / 120, pit], dtype=np.float32)
        info = {}
        return obs, reward, done, info

# === RL BUTTONS ===
train_agent = st.sidebar.button("🚀 Train Agent")
run_simulation = st.sidebar.button("🏁 Run Simulation")

if train_agent:
    st.sidebar.write("Training...")
    env = DummyVecEnv([lambda: F1PitStopEnv()])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    model.save("ppo_f1_pit_agent")
    st.sidebar.success("Training Complete! Model Saved.")

# === SIMULATION ===
if run_simulation:
    st.sidebar.write("Running simulation...")
    env = DummyVecEnv([lambda: F1PitStopEnv()])
    model = PPO.load("ppo_f1_pit_agent")

    obs, _ = env.reset()
    pit_decisions = []

    for lap in range(race_length):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        if int(action) == lap:
            pit_decisions.append(lap)

        if done:
            break

    # === SIMULATED RACE DATA ===
    laps = np.arange(1, race_length + 1)
    lap_times = np.random.normal(90, 3, size=race_length)
    tire_wear = np.maximum(0, 100 - (laps * 1.5))
    fuel_load = np.maximum(0, 100 - (laps * 1.2))

    # === VISUALS ===
    st.subheader("🏁 Race Simulation Results")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', name="Lap Times"))
        fig1.update_layout(title="Lap Times Over Race", template="plotly_dark" if mode == "Dark" else "plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=laps, y=tire_wear, mode='lines+markers', name="Tire Wear"))
        fig2.update_layout(title="Tire Wear Over Race", template="plotly_dark" if mode == "Dark" else "plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=laps, y=fuel_load, mode='lines+markers', name="Fuel Load"))
        fig3.update_layout(title="Fuel Load Over Race", template="plotly_dark" if mode == "Dark" else "plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pit_decisions, y=[pit_stop_time] * len(pit_decisions), mode='markers', marker=dict(size=10, color='red')))
        fig4.update_layout(title="Pit Stop Strategy", xaxis_title="Lap", yaxis_title="Pit Stop Time (s)", template="plotly_dark" if mode == "Dark" else "plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

    st.success(f"Simulation Complete for {selected_driver} at Silverstone! 🏁")

# === FASTF1 REAL TELEMETRY ===
st.header("📡 Live Telemetry & Real Race Replay")

# Enable cache for FastF1
fastf1.Cache.enable_cache('./cache')

year = 2023
gp = 'British Grand Prix'
session_type = 'R'

session = fastf1.get_session(year, gp, session_type)
session.load(telemetry=True, laps=True)

laps_data = session.laps.pick_driver(selected_driver)
telemetry = laps_data.get_car_data().add_distance()

fig_telemetry = go.Figure()
fig_telemetry.add_trace(go.Scatter(x=telemetry['Distance'], y=telemetry['Speed'], mode='lines', name='Speed'))
fig_telemetry.update_layout(title=f"{selected_driver}'s Speed vs Distance at {gp}", xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", template="plotly_dark" if mode == "Dark" else "plotly_white")
st.plotly_chart(fig_telemetry, use_container_width=True)

# === LEADERBOARDS ===
st.header("🏆 Leaderboards")
drivers_leaderboard = pd.DataFrame({
    "Driver": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Carlos Sainz"],
    "Team": ["Red Bull", "Mercedes", "Ferrari", "Ferrari"],
    "Points": [450, 360, 290, 275]
})

teams_leaderboard = pd.DataFrame({
    "Team": ["Red Bull", "Mercedes", "Ferrari", "McLaren"],
    "Points": [720, 600, 520, 430]
})

col1, col2 = st.columns(2)
with col1:
    st.subheader("Drivers")
    st.table(drivers_leaderboard)

with col2:
    st.subheader("Constructors")
    st.table(teams_leaderboard)

# === CIRCUIT PATH ANIMATION ===
st.header("🏎️ Silverstone Circuit Animation (Lap Preview)")

# Dummy simplified Silverstone circuit path
silverstone_coords = [
    (0, 0), (10, 5), (20, 10), (30, 5), (40, 0), (30, -5), (20, -10), (10, -5), (0, 0)
]
circuit_df = pd.DataFrame(silverstone_coords, columns=["x", "y"])

# Generate frames for animation
frames = []
for i in range(1, len(circuit_df) + 1):
    frame = go.Frame(
        data=[
            go.Scatter(
                x=circuit_df["x"][:i],
                y=circuit_df["y"][:i],
                mode="lines+markers",
                line=dict(color="#FF4136", width=4),
                marker=dict(size=12, color="#FFDC00", symbol="circle"),
                name=selected_driver
            )
        ],
        name=f"Lap {i}"
    )
    frames.append(frame)

# Base figure
fig_circuit = go.Figure(
    data=[
        go.Scatter(
            x=[circuit_df["x"][0]],
            y=[circuit_df["y"][0]],
            mode="markers+text",
            marker=dict(size=14, color="green"),
            text=["Start"],
            textposition="top center"
        )
    ],
    layout=go.Layout(
        title="🏁 Silverstone Circuit Animation",
        xaxis=dict(range=[-20, 60], title="X Coordinate"),
        yaxis=dict(range=[-20, 20], title="Y Coordinate"),
        template="plotly_dark" if mode == "Dark" else "plotly_white",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    ),
    frames=frames
)

st.plotly_chart(fig_circuit, use_container_width=True)


# === CIRCUIT BACKGROUND ===
st.markdown("### 🏁 Silverstone Circuit Layout")
st.image("assets/circuits/silverstone_layout.png", use_container_width=True)
