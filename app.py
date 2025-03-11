import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go

# RL Imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# === F1 Race Strategy Simulator === #
st.set_page_config(page_title="🏎️ F1 Race Strategy RL Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - RL Agent + Dynamic Weather + Incidents")
st.markdown("---")

# === TEAM & DRIVER SELECTION === #
st.sidebar.header("🏎️ Team & Driver Selection")
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}
selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
team_colors = teams[selected_team]["color"]
st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.markdown(f"### Base Degradation Factor: {degradation_base}")

# === DRIVER PROFILES ===
driver_profiles = {
    "Lewis Hamilton": {"skill": 0.95, "aggression": 0.4, "wet_skill": 0.9},
    "George Russell": {"skill": 0.90, "aggression": 0.3, "wet_skill": 0.85},
    "Max Verstappen": {"skill": 0.97, "aggression": 0.5, "wet_skill": 0.85},
    "Sergio Perez": {"skill": 0.91, "aggression": 0.35, "wet_skill": 0.80},
    "Charles Leclerc": {"skill": 0.93, "aggression": 0.6, "wet_skill": 0.8},
    "Carlos Sainz": {"skill": 0.92, "aggression": 0.4, "wet_skill": 0.83},
    "Lando Norris": {"skill": 0.89, "aggression": 0.45, "wet_skill": 0.82},
    "Oscar Piastri": {"skill": 0.88, "aggression": 0.38, "wet_skill": 0.81}
}
profile = driver_profiles[selected_driver]

# === DRIVER PHOTO ===
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = 5

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION === #
st.sidebar.header("🚾 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

# === RL ENVIRONMENT === #
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
        self.pit_strategy = []
        obs = np.array([0, 0, 0], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        pit = 1 if action == self.current_lap else 0
        lap_time = 90 + degradation_base * self.current_lap + pit * pit_stop_time
        self.total_time += lap_time
        reward = -lap_time
        self.current_lap += 1
        done = self.current_lap >= race_length

        obs = np.array([self.current_lap / race_length, lap_time / 120, pit], dtype=np.float32)
        info = {}

        return obs, reward, done, info

# === TRAIN RL AGENT === #
train_agent = st.sidebar.button("🚀 Train RL Agent")
run_simulation = st.sidebar.button("🏁 Run Simulation")

if train_agent:
    with st.spinner("Training RL agent..."):
        env = DummyVecEnv([lambda: F1PitStopEnv()])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save("ppo_f1_pit_agent")
    st.sidebar.success("Training Complete! Model Saved.")

# === RUN SIMULATION === #
if run_simulation:
    with st.spinner("Running simulation with trained agent..."):
        env = DummyVecEnv([lambda: F1PitStopEnv()])
        model = PPO.load("ppo_f1_pit_agent")

        obs = env.reset()
        pit_decisions = []

        done = [False]

        for lap in range(race_length):
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)

            done = dones[0]

            if int(action) == lap:
                pit_decisions.append(lap)

            if done:
                break

        # === RACE DATA === #
        def generate_race_data():
            laps = np.arange(1, race_length + 1)
            lap_times = np.random.normal(90, 2, size=race_length)
            lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))
            tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
            fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))
            return laps, lap_times, lead_delta, tire_wear, fuel_load

        laps, lap_times, lead_delta, tire_wear, fuel_load = generate_race_data()

        # === VISUALS === #
        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(
                x=laps,
                y=lap_times,
                mode='lines+markers',
                name='Lap Times',
                line=dict(color=team_colors[0], width=3)
            ))
            fig_lap_times.update_layout(
                title='Lap Times Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Time (s)',
                height=400
            )
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_position_delta = go.Figure()
            fig_position_delta.add_trace(go.Scatter(
                x=laps,
                y=lead_delta,
                mode='lines+markers',
                name='Position Delta',
                line=dict(color=team_colors[1], width=3)
            ))
            fig_position_delta.update_layout(
                title='Track Position Delta Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Time Delta (s)',
                height=400
            )
            st.plotly_chart(fig_position_delta, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(
                x=laps,
                y=tire_wear,
                mode='lines+markers',
                name='Tire Wear (%)',
                line=dict(color='orange', width=3)
            ))
            fig_tire_wear.update_layout(
                title='Tire Wear Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Tire Wear (%)',
                height=400
            )
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        with col4:
            fig_fuel_load = go.Figure()
            fig_fuel_load.add_trace(go.Scatter(
                x=laps,
                y=fuel_load,
                mode='lines+markers',
                name='Fuel Load (%)',
                line=dict(color='yellow', width=3)
            ))
            fig_fuel_load.update_layout(
                title='Fuel Load Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Fuel Load (%)',
                height=400
            )
            st.plotly_chart(fig_fuel_load, use_container_width=True)

        # === PIT STRATEGY VISUAL === #
        st.subheader("🔧 Pit Stop Strategy")
        st.markdown(f"Pit Stops at Laps: {pit_decisions}")

        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(
            x=pit_decisions,
            y=[pit_stop_time for _ in pit_decisions],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Pit Stops'
        ))
        fig_pit.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title='Lap',
            yaxis_title='Pit Stop Time (s)',
            height=400
        )
        st.plotly_chart(fig_pit, use_container_width=True)

        st.sidebar.success("Simulation Complete!")
