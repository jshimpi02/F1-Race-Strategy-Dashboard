import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
from fastf1.core import Laps
from datetime import datetime
import requests
import random
from PIL import Image

# === CONFIG === #
st.set_page_config(page_title="🏎️ F1 Live Race Strategy Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>🏁 F1 Live Race Strategy Dashboard</h1>", unsafe_allow_html=True)

# Enable cache for FastF1
fastf1.Cache.enable_cache('./cache')

# === TEAMS & DRIVERS === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

# === SIDEBAR SELECTIONS === #
st.sidebar.header("🏎️ Team & Driver Selection")
selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
team_colors = teams[selected_team]["color"]

team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)
st.sidebar.markdown(f"### Degradation Factor: `{degradation_base}`")

# === SIMULATION SETTINGS === #
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === WEATHER INTEGRATION === #
def get_weather():
    api_key = "121e2c26dfc4b73738ab60a1f773fc1a"
    city = "Monza"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"] - 273.15
        return f"{desc.title()}, {temp:.1f}°C"
    except:
        return "Clear, 25°C"

weather = get_weather()

# === CIRCUIT BACKGROUND === #
st.markdown("---")
circuit_bg = Image.open("assets/circuits/monza_track.png")
st.image(circuit_bg, caption="Autodromo Nazionale Monza", use_column_width=True)

# === LIVE TELEMETRY FETCH === #
@st.cache_data
def load_telemetry(session_year, gp_name):
    session = fastf1.get_session(session_year, gp_name, 'R')
    session.load()
    laps = session.laps.pick_driver(selected_driver.split()[1][:3].upper())
    return laps

laps_data = load_telemetry(2023, "Monza")

# === LAP TIMES === #
lap_numbers = laps_data['LapNumber'].values
lap_times_sec = laps_data['LapTime'].dt.total_seconds()

# === GENERATE SIMULATION DATA === #
def generate_race_data():
    laps = np.arange(1, race_length + 1)
    lap_times = lap_times_sec if len(lap_times_sec) >= race_length else np.random.normal(90, 2, size=race_length)
    lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))
    tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
    fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))
    return laps, lap_times, lead_delta, tire_wear, fuel_load

laps, lap_times, lead_delta, tire_wear, fuel_load = generate_race_data()

# === MULTI-DRIVER COMPARISON === #
st.markdown(f"<h2 style='color: {team_colors[0]}'>Race Simulation for {selected_driver}</h2>", unsafe_allow_html=True)
st.markdown(f"### Weather: {weather}")

col1, col2 = st.columns(2)

with col1:
    fig_lap_times = go.Figure()
    fig_lap_times.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', name='Lap Times', line=dict(color=team_colors[0])))
    fig_lap_times.update_layout(title='Lap Times', template='plotly_dark')
    st.plotly_chart(fig_lap_times, use_container_width=True)

with col2:
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=laps, y=lead_delta, mode='lines+markers', name='Track Position Delta', line=dict(color=team_colors[1])))
    fig_delta.update_layout(title='Track Position Delta', template='plotly_dark')
    st.plotly_chart(fig_delta, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_tire = go.Figure()
    fig_tire.add_trace(go.Scatter(x=laps, y=tire_wear, mode='lines+markers', name='Tire Wear (%)', line=dict(color='orange')))
    fig_tire.update_layout(title='Tire Wear Over Race', template='plotly_dark')
    st.plotly_chart(fig_tire, use_container_width=True)

with col4:
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(x=laps, y=fuel_load, mode='lines+markers', name='Fuel Load (%)', line=dict(color='yellow')))
    fig_fuel.update_layout(title='Fuel Load Over Race', template='plotly_dark')
    st.plotly_chart(fig_fuel, use_container_width=True)

# === PIT STOP STRATEGY === #
pit_decisions = [random.choice(range(1, race_length)) for _ in range(2)]
st.markdown("## 🔧 Pit Stop Strategy")
st.markdown(f"### Pit Stops at Laps: `{pit_decisions}`")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(x=pit_decisions, y=[pit_stop_time for _ in pit_decisions],
                             mode='markers', marker=dict(size=12, color='red'), name='Pit Stops'))
fig_pit.update_layout(title='Pit Stop Timing', template='plotly_dark')
st.plotly_chart(fig_pit, use_container_width=True)

# === STANDINGS SIMULATION === #
driver_points = random.randint(50, 300)
constructor_points = random.randint(150, 600)

st.sidebar.header("🏆 Championship Standings (Simulated)")
st.sidebar.markdown(f"**Driver Points for {selected_driver}:** {driver_points}")
st.sidebar.markdown(f"**Constructor Points for {selected_team}:** {constructor_points}")

# === FOOTER === #
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: grey;'>© 2025 F1 Race Strategy Dashboard</h4>", unsafe_allow_html=True)
