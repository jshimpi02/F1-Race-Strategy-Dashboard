import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pygad
import fastf1
from fastf1 import plotting

# Enable caching for faster telemetry loading
fastf1.Cache.enable_cache('cache')

# ==== PAGE CONFIG ====
st.set_page_config(page_title="🏎️ F1 Race Strategy - Silverstone GP", layout="wide")

# ==== THEME TOGGLE ====
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)
template = 'plotly_dark' if dark_mode else 'plotly_white'

# ==== TEAM & DRIVER SELECTION ====
st.sidebar.header("🏁 Team & Driver Selection")
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]}
}

selected_team = st.sidebar.selectbox("Select Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])
team_colors = teams[selected_team]["color"]

# Show team logo and driver photo
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_photo_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, width=100)
st.sidebar.image(driver_photo_path, width=100)

# ==== SIMULATION SETTINGS ====
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 52)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (s)", 18, 30, 22)

# ==== GA PARAMETERS ====
num_generations = 50
num_parents_mating = 4
sol_per_pop = 10
num_genes = 3  # number of pit stops

# ==== FITNESS FUNCTION ====
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = sorted(np.clip(solution.astype(int), 1, race_length))
    base_lap_time = 90
    total_time = 0
    degradation = 0.3  # Tire degradation per lap

    last_pit = 0
    for pit in pit_laps:
        stint_length = pit - last_pit
        stint_degradation = degradation * np.sum(np.arange(1, stint_length + 1))
        total_time += base_lap_time * stint_length + stint_degradation
        total_time += pit_stop_time
        last_pit = pit

    # Final stint
    final_stint_length = race_length - last_pit
    final_degradation = degradation * np.sum(np.arange(1, final_stint_length + 1))
    total_time += base_lap_time * final_stint_length + final_degradation

    return -total_time  # Negative because PyGAD maximizes

# ==== RUN GA ====
def run_ga():
    gene_space = [{'low': 1, 'high': race_length} for _ in range(num_genes)]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return sorted(solution.astype(int))

# ==== RUN OPTIMIZATION ====
if st.sidebar.button("Run Pit Stop Optimization"):
    st.info("Running Genetic Algorithm Optimization...")
    best_pit_stops = run_ga()
    st.success(f"Optimal Pit Stops: {best_pit_stops}")
else:
    best_pit_stops = []

# ==== FASTF1 TELEMETRY (Silverstone) ====
session = fastf1.get_session(2023, 'Silverstone', 'R')
session.load()

driver_data = session.laps.pick_driver(selected_driver.split()[1].upper())

track_map = session.get_circuit_info().get('Location', 'Unknown Track')

st.title(f"🏎️ {selected_driver} | {selected_team} - Silverstone GP")
st.markdown(f"**Circuit:** {track_map}")
st.markdown("---")

# ==== CIRCUIT TRACK ANIMATION ====
st.subheader("📍 Circuit Track Map (Silverstone)")

track = session.get_circuit_info()['Layout']
fig_track = go.Figure()

fig_track.add_trace(go.Scatter(
    x=[p[0] for p in track],
    y=[p[1] for p in track],
    mode='lines',
    line=dict(color='white', width=3),
    name='Track Layout'
))

fig_track.add_trace(go.Scatter(
    x=[track[0][0]],
    y=[track[0][1]],
    mode='markers',
    marker=dict(size=10, color='red'),
    name='Start
