import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
from fastf1.core import Laps
from PIL import Image
import time
import os

# Initialize FastF1 cache
if not os.path.exists("cache"):
    os.makedirs("cache")
fastf1.Cache.enable_cache("cache")

# Streamlit page configuration
st.set_page_config(page_title="F1 Race Strategy Simulator", layout="wide")

# === ASSETS ===
TEAM_LOGOS_PATH = "assets/logos/"
DRIVER_HEADSHOTS_PATH = "assets/drivers/"
CIRCUIT_MAP_PATH = "assets/circuits/silverstone.png"

# === PAGE HEADER ===
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/f1_logo.png", width=100)
with col2:
    st.title("🏎️ F1 Race Strategy & Telemetry Dashboard")

st.markdown("---")

# === THEME TOGGLE ===
theme_toggle = st.sidebar.radio("Select Theme", ("Dark Mode", "Light Mode"))
if theme_toggle == "Dark Mode":
    theme_template = "plotly_dark"
else:
    theme_template = "plotly_white"

# === USER SELECTION ===
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": "#00D2BE"},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": "#1E41FF"},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": "#DC0000"},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": "#FF8700"},
}

selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])

# === DISPLAY TEAM LOGO + DRIVER HEADSHOT ===
st.sidebar.image(f"{TEAM_LOGOS_PATH}{selected_team.lower().replace(' ', '_')}.png", use_column_width=True)
st.sidebar.image(f"{DRIVER_HEADSHOTS_PATH}{selected_driver.lower().replace(' ', '_')}.png", use_column_width=True)

# === CIRCUIT INFO ===
gp_year = 2023
gp_name = 'Silverstone'
session_type = 'R'

session = fastf1.get_session(gp_year, gp_name, session_type)
session.load(laps=True, telemetry=True)
circuit_info = session.get_circuit_info()

st.subheader(f"📍 Circuit: {circuit_info['Name']}")
st.markdown(f"**Location:** {circuit_info['Location']}, {circuit_info['Country']}")
st.image(CIRCUIT_MAP_PATH, use_column_width=True)
st.markdown("---")

# === TELEMETRY DATA ===
laps = session.laps.pick_driver(selected_driver.split()[-1].upper())
fastest_lap = laps.pick_fastest()
telemetry = fastest_lap.get_car_data().add_distance()

st.subheader("📈 Telemetry Data (Speed, Throttle, Brake)")

fig_telemetry = go.Figure()

fig_telemetry.add_trace(go.Scatter(
    x=telemetry['Distance'],
    y=telemetry['Speed'],
    mode='lines',
    name='Speed (Km/h)',
    line=dict(color='red')
))

fig_telemetry.add_trace(go.Scatter(
    x=telemetry['Distance'],
    y=telemetry['Throttle'] * 100,
    mode='lines',
    name='Throttle (%)',
    line=dict(color='green')
))

fig_telemetry.add_trace(go.Scatter(
    x=telemetry['Distance'],
    y=telemetry['Brake'] * 100,
    mode='lines',
    name='Brake (%)',
    line=dict(color='blue')
))

fig_telemetry.update_layout(
    template=theme_template,
    xaxis_title='Distance (m)',
    yaxis_title='Telemetry Values',
    height=400
)

st.plotly_chart(fig_telemetry, use_container_width=True)

# === MULTIDRIVER RACE SIMULATION ===
st.subheader("🏁 Race Simulation with Multiple Drivers")

drivers = ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Lando Norris"]
race_laps = 10

race_data = pd.DataFrame({
    "Lap": list(range(1, race_laps + 1)),
})

for driver in drivers:
    lap_times = np.cumsum(np.random.uniform(80, 90, race_laps))
    race_data[driver] = lap_times

fig_race = go.Figure()

for driver in drivers:
    fig_race.add_trace(go.Scatter(
        x=race_data["Lap"],
        y=race_data[driver],
        mode='lines+markers',
        name=driver
    ))

fig_race.update_layout(
    title="Driver Lap Times",
    template=theme_template,
    xaxis_title="Lap",
    yaxis_title="Cumulative Lap Time (s)",
    height=500
)

st.plotly_chart(fig_race, use_container_width=True)

# === LEADERBOARD ===
st.subheader("🏆 Driver Leaderboard")
leaderboard = pd.DataFrame({
    "Driver": drivers,
    "Team": ["Mercedes", "Red Bull Racing", "Ferrari", "McLaren"],
    "Points": [312, 350, 260, 280]
}).sort_values(by="Points", ascending=False)

for idx, row in leaderboard.iterrows():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image(f"{DRIVER_HEADSHOTS_PATH}{row['Driver'].lower().replace(' ', '_')}.png", width=50)
    with col2:
        st.markdown(f"**{row['Driver']}** ({row['Team']})")
    with col3:
        st.markdown(f"🏆 {row['Points']} pts")

# === CIRCUIT TRACK ANIMATION (SIMPLIFIED DEMO) ===
st.subheader("🏟️ Circuit Animation")

track_x = telemetry['X']
track_y = telemetry['Y']

fig_circuit = go.Figure()

fig_circuit.add_trace(go.Scatter(
    x=track_x,
    y=track_y,
    mode='lines',
    name='Track Layout',
    line=dict(color='white', width=2)
))

fig_circuit.add_trace(go.Scatter(
    x=[track_x.iloc[0]],
    y=[track_y.iloc[0]],
    mode='markers',
    marker=dict(size=15, color='red'),
    name='Car Start'
))

fig_circuit.update_layout(
    template=theme_template,
    showlegend=False,
    height=600,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig_circuit, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.caption("© 2025 F1 Race Strategy Simulator | Powered by Streamlit, Plotly, FastF1")

