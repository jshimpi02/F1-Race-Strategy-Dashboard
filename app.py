# ==== IMPORTS ====
import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import fastf1
from fastf1.api import Cache
import plotly.graph_objects as go
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# ==== ENABLE CACHE ====
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# ==== PAGE CONFIG ====
st.set_page_config(page_title="🏎️ F1 Race Strategy & Telemetry Dashboard", layout="wide")

# ==== THEME TOGGLE ====
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", True)
theme_template = 'plotly_dark' if dark_mode else 'plotly_white'
st.markdown(f"<style>body {{ background-color: {'#0e1117' if dark_mode else '#ffffff'} }}</style>", unsafe_allow_html=True)

# ==== HEADER ====
st.sidebar.image("assets/f1_logo.png", width=150)
st.title("🏎️ F1 Strategy + Telemetry + Leaderboards")
st.markdown("---")

# ==== TEAMS & DRIVERS ====
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]}
}

selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
team_colors = teams[selected_team]["color"]

st.sidebar.image(f"assets/logos/{selected_team.lower().replace(' ', '_')}.png", caption=selected_team, use_container_width=True)
st.sidebar.image(f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png", caption=selected_driver, use_container_width=True)

# ==== SIMULATION SETTINGS ====
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
num_opponents = st.sidebar.slider("Number of Opponents", 3, 10, 5)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# ==== FASTF1 SESSION ====
session = fastf1.get_session(2023, 'Silverstone', 'R')
session.load(laps=True, telemetry=True)

driver_abbr = selected_driver.split()[-1].upper()[:3]
laps_data = session.laps.pick_driver(driver_abbr)
telemetry_data = laps_data.iloc[0].get_car_data().add_distance()

x_coords = telemetry_data['X']
y_coords = telemetry_data['Y']

# ==== CIRCUIT MAP ====
st.subheader("📍 Silverstone Circuit Track Map")
fig_circuit = go.Figure()

# Add Circuit Image
fig_circuit.add_layout_image(dict(
    source="assets/silverstone_layout.png",
    x=0,
    y=0,
    sizex=500,
    sizey=500,
    xref="x",
    yref="y",
    opacity=0.5,
    layer="below"
))

# Driver Telemetry Path
fig_circuit.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="lines", line=dict(color='cyan', width=2)))
fig_circuit.add_trace(go.Scatter(x=[x_coords.iloc[0]], y=[y_coords.iloc[0]], mode="markers", marker=dict(size=10, color="red")))

fig_circuit.update_layout(title="Telemetry Lap Path", xaxis=dict(visible=False), yaxis=dict(visible=False), height=500, template=theme_template)
st.plotly_chart(fig_circuit, use_container_width=True)

# ==== RL ENVIRONMENT ====
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
        lap_time = 90 + pit * pit_stop_time
        self.total_time += lap_time
        reward = -lap_time
        self.current_lap += 1
        done = self.current_lap >= race_length

        obs = np.array([self.current_lap / race_length, lap_time / 120, pit], dtype=np.float32)
        info = {}
        return obs, reward, done, info

# ==== TRAIN & SIMULATE ====
train_agent = st.sidebar.button("🚀 Train RL Agent")
run_simulation = st.sidebar.button("🏁 Run Simulation")

if train_agent:
    with st.spinner("Training RL agent..."):
        env = DummyVecEnv([lambda: F1PitStopEnv()])
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=5000)
        model.save("ppo_f1_pit_agent")
    st.sidebar.success("Training Complete! ✅")

# ==== MULTI-DRIVER SIMULATION ====
def simulate_opponents(num_drivers, race_length):
    positions = np.arange(1, num_drivers + 1)
    lap_times = np.random.normal(90, 5, size=(num_drivers, race_length))
    return positions, lap_times

if run_simulation:
    with st.spinner("Running Race Simulation..."):
        env = DummyVecEnv([lambda: F1PitStopEnv()])
        model = PPO.load("ppo_f1_pit_agent")

        obs = env.reset()
        pit_decisions = []
        done = [False]

        driver_lap_times = []
        for lap in range(race_length):
            action, _ = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            lap_time = 90 + pit_stop_time if int(action) == lap else 90
            driver_lap_times.append(lap_time)

            if int(action) == lap:
                pit_decisions.append(lap)

            if dones[0]:
                break

        # Opponent Simulation
        positions, opponent_laps = simulate_opponents(num_opponents, race_length)

        col1, col2 = st.columns(2)

        with col1:
            fig_laps = go.Figure()
            fig_laps.add_trace(go.Scatter(y=driver_lap_times, x=np.arange(1, race_length + 1), mode='lines+markers', name=selected_driver, line=dict(color=team_colors[0])))
            for idx in range(num_opponents):
                fig_laps.add_trace(go.Scatter(y=opponent_laps[idx], x=np.arange(1, race_length + 1), mode='lines', name=f"Driver {idx+1}"))

            fig_laps.update_layout(title="Lap Times Comparison", template=theme_template)
            st.plotly_chart(fig_laps, use_container_width=True)

        with col2:
            fig_positions = go.Figure()
            total_times = [sum(driver_lap_times)] + [sum(opponent_laps[idx]) for idx in range(num_opponents)]
            sorted_times = sorted(zip(total_times, [selected_driver] + [f"Driver {idx+1}" for idx in range(num_opponents)]))
            drivers_sorted = [d for _, d in sorted_times]
            times_sorted = [t for t, _ in sorted_times]

            fig_positions.add_trace(go.Bar(y=drivers_sorted, x=times_sorted, orientation='h', marker_color='cyan'))
            fig_positions.update_layout(title="Final Race Positions (Time)", template=theme_template)
            st.plotly_chart(fig_positions, use_container_width=True)

        st.success("Simulation Complete!")

# ==== LEADERBOARDS ====
st.markdown("---")
st.subheader("🏆 2025 F1 Leaderboards")

# Sample Leaderboard Data
drivers_standings = pd.DataFrame({
    "Driver": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Lando Norris", "Carlos Sainz"],
    "Team": ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren", "Ferrari"],
    "Points": [310, 290, 280, 250, 240]
})

constructors_standings = pd.DataFrame({
    "Team": ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren"],
    "Points": [600, 550, 500, 450]
})

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👨‍💼 Drivers Championship")
    for idx, row in drivers_standings.iterrows():
        st.image(f"assets/drivers/{row['Driver'].lower().replace(' ', '_')}.png", width=50)
        st.write(f"**{idx+1}. {row['Driver']}** ({row['Team']}) - {row['Points']} pts")

with col2:
    st.markdown("### 🏢 Constructors Championship")
    for idx, row in constructors_standings.iterrows():
        st.image(f"assets/logos/{row['Team'].lower().replace(' ', '_')}.png", width=50)
        st.write(f"**{idx+1}. {row['Team']}** - {row['Points']} pts")

# ==== FOOTER ====
st.markdown("---")
st.markdown("Made for F1 Race Simulations & Analytics ❤️")
