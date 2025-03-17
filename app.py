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

circuit_info = session.get_circuit_info()

circuit_info = session.get_circuit_info()

track_name = getattr(circuit_info, 'name', 'Unknown Track')
track_location = getattr(circuit_info, 'location', 'Unknown Location')
track_country = getattr(circuit_info, 'country', 'Unknown Country')

st.title(f"🏎️ {selected_driver} | {selected_team} - {track_name} GP")
st.markdown(f"**Circuit:** {track_name} | Location: {track_location}, {track_country}")
st.markdown("---")


# ==== CIRCUIT TRACK ANIMATION ====
st.subheader("📍 Circuit Track Map (Silverstone)")

# Simulated dummy track layout (you can replace this with actual track points)
track_x = [0, 1, 2, 3, 4, 3, 2, 1, 0]
track_y = [0, 1, 2, 3, 4, 5, 6, 5, 4]

fig_track = go.Figure()

# Plot the track line
fig_track.add_trace(go.Scatter(
    x=track_x,
    y=track_y,
    mode='lines+markers',
    name='Circuit Path',
    line=dict(color='white', width=3)
))

fig_track.update_layout(
    title='Silverstone Circuit Layout (Simulated)',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    height=500
)

st.plotly_chart(fig_track, use_container_width=True)


# ==== SIMULATION DATA (Mockup based on settings) ====
laps = np.arange(1, race_length + 1)
lap_times = np.random.normal(90, 1.5, size=race_length)
tire_wear = np.clip(100 - (laps * 1.5), 0, 100)
fuel_load = np.clip(100 - (laps * 2), 0, 100)

# ==== LAP TIMES & TELEMETRY ====
st.subheader("📊 Lap Times & Telemetry")

col1, col2 = st.columns(2)

with col1:
    fig_lap_times = go.Figure()
    fig_lap_times.add_trace(go.Scatter(
        x=laps,
        y=lap_times,
        mode='lines+markers',
        name='Lap Time',
        line=dict(color=team_colors[0], width=3)
    ))
    fig_lap_times.update_layout(
        title='Lap Times',
        template=template,
        height=400,
        xaxis_title='Lap',
        yaxis_title='Time (s)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_lap_times, use_container_width=True)

with col2:
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(
        x=laps,
        y=fuel_load,
        mode='lines+markers',
        name='Fuel Load',
        line=dict(color='yellow', width=3)
    ))
    fig_fuel.update_layout(
        title='Fuel Load Over Race',
        template=template,
        height=400,
        xaxis_title='Lap',
        yaxis_title='Fuel (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

# ==== TIRE WEAR ====
st.subheader("🛞 Tire Wear Progression")

fig_tire = go.Figure()
fig_tire.add_trace(go.Scatter(
    x=laps,
    y=tire_wear,
    mode='lines+markers',
    name='Tire Wear',
    line=dict(color='orange', width=3)
))
fig_tire.update_layout(
    template=template,
    height=400,
    xaxis_title='Lap',
    yaxis_title='Tire Wear (%)',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_tire, use_container_width=True)

# ==== PIT STRATEGY ====
st.subheader("🔧 Pit Stop Strategy")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=best_pit_stops if best_pit_stops else [],
    y=[pit_stop_time] * len(best_pit_stops),
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stop'
))
fig_pit.update_layout(
    template=template,
    height=400,
    xaxis_title='Lap',
    yaxis_title='Pit Stop Time (s)',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_pit, use_container_width=True)

# ==== LEADERBOARD ====
st.subheader("🏆 Driver Standings (Mockup)")

driver_standings = pd.DataFrame({
    "Driver": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "George Russell", "Lando Norris"],
    "Team": ["Red Bull Racing", "Mercedes", "Ferrari", "Mercedes", "McLaren"],
    "Points": [290, 230, 210, 190, 185]
})

st.table(driver_standings)

st.sidebar.success("Simulation & Telemetry Loaded Successfully!")

