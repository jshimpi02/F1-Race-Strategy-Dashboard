import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import plotly.graph_objects as go
import fastf1

# === CONFIG === #
st.set_page_config(page_title="🏁 F1 Race Strategy Dashboard", layout="wide")

# === DARK/LIGHT MODE === #
theme_mode = st.sidebar.radio("Select Mode", ["Dark", "Light"])
if theme_mode == "Dark":
    plotly_theme = "plotly_dark"
    background_color = "#111"
else:
    plotly_theme = "plotly_white"
    background_color = "#f5f5f5"

# === SILVERSTONE BACKGROUND === #
circuit_bg = "assets/circuits/silverstone.png"

# === HEADER === #
st.markdown(f"""
    <div style="background-image: url('{circuit_bg}'); 
                background-size: cover; 
                padding: 50px; 
                text-align: center; 
                color: white; 
                font-size: 36px;
                font-weight: bold;">
        F1 Race Strategy & Telemetry Dashboard - Silverstone GP 🏎️
    </div>
""", unsafe_allow_html=True)

st.sidebar.image("assets/f1_logo.png", width=150)

# === TEAMS & DRIVERS (2025 SEASON) === #
teams_2025 = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]},
    "Aston Martin": {"drivers": ["Fernando Alonso", "Lance Stroll"], "color": ["#006F62", "#FFFFFF"]},
}

# === TEAM/DRIVER SELECTION === #
selected_team = st.sidebar.selectbox("Select Your Team", list(teams_2025.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams_2025[selected_team]["drivers"])

team_colors = teams_2025[selected_team]["color"]
team_logo = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_photo = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_photo, caption=selected_driver, use_container_width=True)

# === SIM SETTINGS === #
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === DRIVER & TEAM LEADERBOARDS === #
st.subheader("🏆 2025 Driver & Constructor Standings")

driver_leaderboard = pd.DataFrame({
    "Driver": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris", "Fernando Alonso"],
    "Team": ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Aston Martin"],
    "Points": [320, 285, 275, 260, 240]
})

constructor_leaderboard = pd.DataFrame({
    "Team": ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Aston Martin"],
    "Points": [605, 540, 510, 490, 470]
})

col1, col2 = st.columns(2)
col1.table(driver_leaderboard)
col2.table(constructor_leaderboard)

# === TELEMETRY DATA SIMULATION === #
st.markdown("### 📡 Live Telemetry Feed")

def generate_telemetry_data(lap):
    speed = random.randint(280, 320)
    gear = random.choice([6, 7, 8])
    throttle = random.randint(80, 100)
    brake = random.randint(0, 20)
    return speed, gear, throttle, brake

telemetry_placeholder = st.empty()

# === PIT STOP STRATEGY GENETIC ALGO === #
def simulate_strategy(pit_laps):
    total_time = 0
    for lap in range(1, race_length + 1):
        base_time = 90 + (lap * 0.2)
        if lap in pit_laps:
            total_time += base_time + pit_stop_time
        else:
            total_time += base_time
    return total_time

best_pit_strategy = [15, 35]

# === PROGRESS BAR === #
progress_bar = st.progress(0)
progress_text = st.empty()

lap_times = []
lead_deltas = []
tire_wear = []
fuel_load = []
pit_stops = []

for lap in range(1, race_length + 1):
    progress = lap / race_length
    progress_bar.progress(progress)
    progress_text.text(f"Lap {lap}/{race_length} in progress...")

    # Telemetry update
    speed, gear, throttle, brake = generate_telemetry_data(lap)
    telemetry_placeholder.metric("Speed (km/h)", speed, delta=None)
    telemetry_placeholder.metric("Gear", gear)
    telemetry_placeholder.metric("Throttle (%)", throttle)
    telemetry_placeholder.metric("Brake (%)", brake)

    # Race data
    lap_time = 90 + random.uniform(-1, 1) + lap * 0.2
    lap_times.append(lap_time)
    lead_deltas.append(sum(random.choices([-0.5, 0, 0.5], k=lap)))
    tire_wear.append(max(0, 100 - (lap * 1.5)))
    fuel_load.append(max(0, 100 - (lap * (100 / race_length))))

    if lap in best_pit_strategy:
        pit_stops.append(lap)

    time.sleep(0.1)

st.success("Race Simulation Complete! ✅")

# === GRAPHS === #
st.subheader("📊 Race Performance Charts")

laps = np.arange(1, race_length + 1)

col3, col4 = st.columns(2)

with col3:
    fig1 = go.Figure(go.Scatter(x=laps, y=lap_times, mode="lines+markers", line=dict(color=team_colors[0])))
    fig1.update_layout(title="Lap Times", template=plotly_theme, xaxis_title="Lap", yaxis_title="Time (s)", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    fig2 = go.Figure(go.Scatter(x=laps, y=lead_deltas, mode="lines+markers", line=dict(color=team_colors[1])))
    fig2.update_layout(title="Lead Delta", template=plotly_theme, xaxis_title="Lap", yaxis_title="Delta (s)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    fig3 = go.Figure(go.Scatter(x=laps, y=tire_wear, mode="lines+markers", line=dict(color='orange')))
    fig3.update_layout(title="Tire Wear (%)", template=plotly_theme, xaxis_title="Lap", yaxis_title="Tire Wear", height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col6:
    fig4 = go.Figure(go.Scatter(x=laps, y=fuel_load, mode="lines+markers", line=dict(color='yellow')))
    fig4.update_layout(title="Fuel Load (%)", template=plotly_theme, xaxis_title="Lap", yaxis_title="Fuel Load", height=400)
    st.plotly_chart(fig4, use_container_width=True)

# === PIT STOP STRATEGY VISUAL === #
st.subheader("🔧 Pit Stop Strategy")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=pit_stops,
    y=[pit_stop_time] * len(pit_stops),
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))
fig_pit.update_layout(
    title="Pit Stops Over Race",
    template=plotly_theme,
    xaxis_title="Lap",
    yaxis_title="Pit Stop Time (s)",
    height=400
)
st.plotly_chart(fig_pit, use_container_width=True)

st.sidebar.success("Dashboard Complete!")
