import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
import pygad
import random
import datetime

# Set Streamlit page config
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# === DARK/LIGHT MODE TOGGLE === #
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# === STYLING BASED ON MODE === #
plotly_template = "plotly_dark" if dark_mode else "plotly_white"
background_color = "#0e1117" if dark_mode else "#ffffff"
font_color = "white" if dark_mode else "black"

# === HEADER SECTION === #
st.markdown(f"<h1 style='text-align: center; color: {font_color};'>🏎️ F1 Race Strategy Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# === CACHE ENABLE === #
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# === TEAM & DRIVER SELECTION === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

st.sidebar.image("assets/f1_logo.png", width=150)

selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

# Display Team Logo & Driver Photo
st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

degradation_base = teams[selected_team]["degradation_factor"]
team_colors = teams[selected_team]["color"]

# === SESSION & CIRCUIT INFO === #
gp_year = 2023
gp_name = 'Silverstone'
session_type = 'R'

session = fastf1.get_session(gp_year, gp_name, session_type)
session.load(laps=True, telemetry=True)

circuit_info = session.get_circuit_info()

st.subheader(f"📍 Circuit: {circuit_info.get('Name', 'Unknown Circuit')}")
st.markdown(f"**Location:** {circuit_info.get('Location', 'Unknown Location')}, {circuit_info.get('Country', 'Unknown Country')}")
st.markdown(f"**Length:** {circuit_info.get('Length', 'Unknown Length')} m")

# === CIRCUIT ANIMATION === #
st.subheader("📍 Silverstone Track Map")
track_map = plotting.get_circuit_map(gp_name)

fig_track = go.Figure()

fig_track.add_trace(go.Scatter(
    x=track_map.x,
    y=track_map.y,
    mode='lines',
    line=dict(color='white', width=3),
    name='Track Layout'
))

fig_track.update_layout(
    template=plotly_template,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    width=700,
    height=600,
    showlegend=False
)

st.plotly_chart(fig_track, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = st.sidebar.slider("Number of Opponent Drivers", 1, 10, 5)

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION === #
st.sidebar.header("🚾 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

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

# === GENETIC ALGORITHM FOR PIT STRATEGY === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = [i for i, pit in enumerate(solution) if pit == 1]
    total_time = 0
    tire_wear = 0
    for lap in range(race_length):
        lap_time = 90 + degradation_base * lap + tire_wear
        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 0
        else:
            tire_wear += degradation_base
        total_time += lap_time
    return -total_time

def run_ga():
    num_genes = race_length
    gene_space = [0, 1]

    ga_instance = pygad.GA(
        num_generations=50,
        sol_per_population=10,
        num_parents_mating=5,
        num_genes=num_genes,
        fitness_func=fitness_func,
        gene_space=gene_space
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    pit_laps = [i for i, pit in enumerate(solution) if pit == 1]
    return pit_laps

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Optimize Pit Stops (GA)"):
    with st.spinner("Running Genetic Algorithm..."):
        best_pit_stops = run_ga()
        st.success(f"Pit Stop Strategy Generated ✅: {best_pit_stops}")
else:
    best_pit_stops = []

# === GENERATE RACE DATA === #
def generate_race_data():
    laps = np.arange(1, race_length + 1)
    lap_times = np.random.normal(90, 2, size=race_length)
    lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))
    tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
    fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))
    return laps, lap_times, lead_delta, tire_wear, fuel_load

laps, lap_times, lead_delta, tire_wear, fuel_load = generate_race_data()

# === PLOTLY CHARTS === #
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
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
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
        name='Track Position Delta',
        line=dict(color=team_colors[1], width=3)
    ))
    fig_position_delta.update_layout(
        title='Track Position Delta Over Race',
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
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
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
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
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
        xaxis_title='Lap',
        yaxis_title='Fuel Load (%)',
        height=400
    )
    st.plotly_chart(fig_fuel_load, use_container_width=True)

# === PIT STRATEGY VISUAL === #
st.subheader("🔧 Pit Stop Strategy")
st.markdown(f"Pit Stops at Laps: {best_pit_stops}")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=best_pit_stops,
    y=[pit_stop_time for _ in best_pit_stops],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))
fig_pit.update_layout(
    template=plotly_template,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    font=dict(color=font_color),
    xaxis_title='Lap',
    yaxis_title='Pit Stop Time (s)',
    height=400
)
st.plotly_chart(fig_pit, use_container_width=True)

# === FOOTER === #
st.sidebar.markdown("---")
st.sidebar.info("Developed for F1 Strategy Simulation Demo 🚀")
