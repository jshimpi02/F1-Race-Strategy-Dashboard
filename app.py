import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# RL Imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# === PHASE 7: F1 Race Strategy Simulator ===
st.set_page_config(page_title="🏎️ F1 Race Strategy RL Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - RL Agent + Dynamic Weather + Incidents")
st.markdown("---")

# === TEAM & DRIVER SELECTION ===
st.sidebar.header("🏎️ Team & Driver Selection")
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/8/89/Mercedes-Benz_in_Motorsport_logo.svg/1920px-Mercedes-Benz_in_Motorsport_logo.svg.png"},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/0/01/Red_Bull_Racing_logo.svg/1920px-Red_Bull_Racing_logo.svg.png"},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/3/3d/Scuderia_Ferrari_Logo.svg/1920px-Scuderia_Ferrari_Logo.svg.png"},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/1/19/McLaren_Racing_logo.svg/1920px-McLaren_Racing_logo.svg.png"}
}
selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
team_logo = teams[selected_team]["logo"]
st.sidebar.image(team_logo, caption=selected_team, use_container_width=True)
st.sidebar.markdown(f"### Base Degradation Factor: {degradation_base}")

# === DRIVER PROFILES ===
driver_profiles = {
    "Lewis Hamilton": {"skill": 0.95, "aggression": 0.4, "wet_skill": 0.9, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Lewis_Hamilton_2022.jpg/440px-Lewis_Hamilton_2022.jpg"},
    "George Russell": {"skill": 0.90, "aggression": 0.3, "wet_skill": 0.85, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/George_Russell_2021.jpg/440px-George_Russell_2021.jpg"},
    "Max Verstappen": {"skill": 0.97, "aggression": 0.5, "wet_skill": 0.85, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Max_Verstappen_2017_Malaysia_3.jpg/440px-Max_Verstappen_2017_Malaysia_3.jpg"},
    "Sergio Perez": {"skill": 0.91, "aggression": 0.35, "wet_skill": 0.80, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Sergio_P%C3%A9rez_2022.jpg/440px-Sergio_P%C3%A9rez_2022.jpg"},
    "Charles Leclerc": {"skill": 0.93, "aggression": 0.6, "wet_skill": 0.8, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Charles_Leclerc_2019.jpg/440px-Charles_Leclerc_2019.jpg"},
    "Carlos Sainz": {"skill": 0.92, "aggression": 0.4, "wet_skill": 0.83, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Carlos_Sainz_2021.jpg/440px-Carlos_Sainz_2021.jpg"},
    "Lando Norris": {"skill": 0.89, "aggression": 0.45, "wet_skill": 0.82, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Lando_Norris_2021.jpg/440px-Lando_Norris_2021.jpg"},
    "Oscar Piastri": {"skill": 0.88, "aggression": 0.38, "wet_skill": 0.81, "photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Oscar_Piastri_2023.jpg/440px-Oscar_Piastri_2023.jpg"}
}
profile = driver_profiles[selected_driver]

# === DRIVER PHOTO ===
st.sidebar.image(profile["photo"], caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS ===
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = 5

# === WEATHER SETTINGS ===
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION ===
st.sidebar.header("🛞 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

# (Leaving the remaining parts of the code as they were, focusing only on adding the logos/photos/sidebar visuals as requested)

# === Define Gym Environment ===
class F1PitStopEnv(gym.Env):
    def __init__(self):
        super(F1PitStopEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lap = 1
        self.tire_wear = 0
        self.weather = "Clear"
        self.grip = 1.0
        self.total_time = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        weather_map = {"Clear": 0, "Light Rain": 1, "Heavy Rain": 2}
        return np.array([
            self.lap,
            self.tire_wear,
            weather_map[self.weather],
            self.grip
        ], dtype=np.float32)

    def step(self, action):
        pit_penalty = 0
        if self.lap % 10 == 0:
            self.weather = np.random.choice(["Clear", "Light Rain", "Heavy Rain"], p=[0.5, 0.3, 0.2])

        self.grip = {"Clear": 1.0, "Light Rain": 0.8, "Heavy Rain": 0.6}[self.weather]

        if action == 1:
            pit_penalty = pit_stop_time
            self.tire_wear = 0
        else:
            self.tire_wear += degradation_base + tire_options[selected_tire]

        lap_time = 100 + (self.tire_wear * self.lap) + (1 - self.grip) * 20 + pit_penalty
        self.total_time += lap_time

        reward = -lap_time
        self.lap += 1

        terminated = self.lap > race_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

# === Train the RL Agent ===
st.sidebar.markdown("---")
train_rl = st.sidebar.button("Train RL Pit Strategy Agent")

if train_rl:
    st.write("### Training RL Pit Strategy Agent...")
    env = DummyVecEnv([lambda: F1PitStopEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_f1_pit_agent")
    st.success("Training Complete! Model saved as ppo_f1_pit_agent.")

# === RL Agent Decision Function ===
def rl_agent_decision(model, lap, tire_wear, weather, grip):
    weather_map = {"Clear": 0, "Light Rain": 1, "Heavy Rain": 2}
    obs = np.array([[lap, tire_wear, weather_map[weather], grip]], dtype=np.float32)
    action, _states = model.predict(obs)
    return action == 1

# === RUN SIMULATION WITH RL AGENT ===
if st.sidebar.button("Run RL Race Simulation 🚀"):
    st.subheader(f"RL Race Simulation: {selected_driver} for {selected_team}")
    st.markdown("---")

    MODEL_PATH = "models/ppo_f1_pit_agent.zip"
    model = PPO.load(MODEL_PATH)

    player_tire = selected_tire
    player_total_time = 0
    player_lap_times = []
    pit_laps = []

    tire_degradation = {
        "Soft": 0.40,
        "Medium": 0.25,
        "Hard": 0.15,
        "Intermediates": 0.30,
        "Wets": 0.35
    }
    final_degradation = degradation_base + tire_degradation[player_tire]

    dynamic_weather = []
    tire_wear_history = []
    fuel_load_history = []
    fuel_load = 100

    for lap in range(1, race_length + 1):
        if selected_weather == "Dynamic Weather":
            if lap == 1 or lap % 15 == 0:
                current_weather = np.random.choice(["Clear", "Light Rain", "Heavy Rain"], p=[0.5, 0.3, 0.2])
            dynamic_weather.append(current_weather)
        else:
            current_weather = selected_weather
            dynamic_weather.append(current_weather)

        grip_level = {"Clear": 1.0, "Light Rain": 0.8, "Heavy Rain": 0.6}[current_weather]
        grip_penalty = (1 - grip_level) * 20

        pit_decision = rl_agent_decision(model, lap, lap * final_degradation, current_weather, grip_level)

        if pit_decision:
            pit_laps.append(lap)
            final_degradation = degradation_base + tire_degradation[random.choice(list(tire_degradation.keys()))]
            lap_time = 100 + pit_stop_time + grip_penalty
        else:
            lap_time = 100 + (lap * final_degradation) + grip_penalty

        player_total_time += lap_time
        player_lap_times.append(lap_time)

        tire_wear_history.append(lap * final_degradation)
        fuel_load -= 100 / race_length
        fuel_load_history.append(fuel_load)

    laps = np.arange(1, race_length + 1)

    # === Lap Times Plot ===
    st.write("### RL Agent Lap Times")
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(laps, player_lap_times, label=f'{selected_driver} (RL Agent)', linewidth=2, color='blue')
    ax1.scatter(pit_laps, [player_lap_times[i - 1] for i in pit_laps], color='red', label='Pit Stops', zorder=5)
    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Lap Time (seconds)")
    ax1.set_title("RL Agent Lap Times with Pit Stops")
    ax1.legend()
    st.pyplot(fig1)

    # === Tire Wear Plot ===
    st.write("### Tire Wear Over Race")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(laps, tire_wear_history, color='orange')
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Tire Wear")
    ax2.set_title("Tire Wear Progression During the Race")
    st.pyplot(fig2)

    # === Fuel Load Plot ===
    st.write("### Fuel Load Over Race")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(laps, fuel_load_history, color='green')
    ax3.set_xlabel("Lap")
    ax3.set_ylabel("Fuel Remaining (%)")
    ax3.set_title("Fuel Load Progression During the Race")
    st.pyplot(fig3)

    st.success("RL Race Complete! Ready for next simulation 🏁")
