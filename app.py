import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import requests
import time

# === STREAMLIT PAGE SETUP === #
st.set_page_config(page_title="🏎️ F1 Race Strategy RL Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - RL Agent + Dynamic Weather + Incidents + Telemetry")
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

# === DRIVER PROFILES === #
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

# === DRIVER PHOTO === #
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

# === TELEMETRY DATA FETCHING === #
def fetch_live_telemetry(driver="Max Verstappen"):
    # Simulated for now, can be hooked to Ergast API or real F1 source
    return {
        "speed": random.randint(280, 350),
        "gear": random.randint(5, 8),
        "rpm": random.randint(10000, 15000),
        "throttle": random.randint(70, 100),
        "brake": random.randint(0, 30)
    }

# === RACE SIMULATION === #
st.header("🏁 Live Race Simulation")
laps = np.arange(1, race_length + 1)

speed_list = []
throttle_list = []
brake_list = []

progress = st.progress(0, text=f"Starting race for {selected_driver}")
status_placeholder = st.empty()

for lap in range(1, race_length + 1):
    telemetry = fetch_live_telemetry(selected_driver)

    speed_list.append(telemetry["speed"])
    throttle_list.append(telemetry["throttle"])
    brake_list.append(telemetry["brake"])

    progress.progress(lap / race_length, text=f"Lap {lap}/{race_length}")
    status_placeholder.info(f"Lap {lap}: Speed {telemetry['speed']} km/h | Gear {telemetry['gear']} | RPM {telemetry['rpm']}")
    time.sleep(0.05)

# === LIVE TELEMETRY GRAPHS === #
col1, col2 = st.columns(2)

with col1:
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=laps, y=speed_list, mode='lines+markers', name='Speed (km/h)', line=dict(color='cyan')))
    fig_speed.update_layout(title="Speed over Laps", template="plotly_dark", height=400)
    st.plotly_chart(fig_speed, use_container_width=True)

with col2:
    fig_throttle_brake = go.Figure()
    fig_throttle_brake.add_trace(go.Scatter(x=laps, y=throttle_list, mode='lines', name='Throttle (%)', line=dict(color='green')))
    fig_throttle_brake.add_trace(go.Scatter(x=laps, y=brake_list, mode='lines', name='Brake (%)', line=dict(color='red')))
    fig_throttle_brake.update_layout(title="Throttle & Brake over Laps", template="plotly_dark", height=400)
    st.plotly_chart(fig_throttle_brake, use_container_width=True)

# === PIT STRATEGY PLACEHOLDER === #
st.subheader("🔧 Pit Stop Strategy")
pit_decisions = [random.choice(laps.tolist()) for _ in range(2)]
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
