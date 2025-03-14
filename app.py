import os
os.environ["PYTORCH_JIT"] = "0"

# Ensure cache directory exists
if not os.path.exists('./cache'):
    os.makedirs('./cache')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygad
import fastf1
from fastf1 import plotting
from datetime import datetime

# Enable FastF1 cache
fastf1.Cache.enable_cache('./cache')

# ============================ STREAMLIT CONFIG ============================
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator + Telemetry + GA Optimization")
st.markdown("---")

# ============================ TEAM & DRIVER SELECTION ============================
st.sidebar.header("🏎️ Team & Driver Selection")

teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

selected_team = st.sidebar.selectbox("Select Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
team_colors = teams[selected_team]["color"]

team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# ============================ SIMULATION SETTINGS ============================
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# ============================ TIRE COMPOUND SELECTION ============================
st.sidebar.header("🚾 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

# ============================ RACE LOGIC FUNCTIONS ============================
def get_weather_factor(weather):
    if weather == "Clear":
        return 1.0
    elif weather == "Light Rain":
        return 1.1
    elif weather == "Heavy Rain":
        return 1.25
    elif weather == "Dynamic Weather":
        return np.random.choice([1.0, 1.1, 1.25])
    return 1.0

def simulate_race(pit_laps):
    lap_times = []
    tire_wear = 100.0
    fuel_load = 100.0
    current_weather = get_weather_factor(selected_weather)

    for lap in range(1, race_length + 1):
        degradation = degradation_base * (lap if lap <= race_length else race_length)
        tire_wear = max(0, tire_wear - degradation * 100)
        fuel_load = max(0, fuel_load - (100 / race_length))

        base_time = 90
        lap_time = base_time + degradation * 20 + current_weather * 5 + (100 - tire_wear) * 0.1
        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 100  # reset tire after pit stop

        lap_times.append(lap_time)
    return np.sum(lap_times), lap_times

# ============================ GENETIC ALGORITHM FITNESS ============================
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = [int(lap) for lap in solution if 1 <= lap <= race_length]
    total_time, _ = simulate_race(pit_laps)
    return -total_time  # minimize total race time

# ============================ RUN GA OPTIMIZATION ============================
def run_ga():
    st.info("Running Genetic Algorithm to optimize pit stops...")
    num_generations = 50
    num_parents_mating = 5
    sol_per_pop = 10
    num_genes = 3  # Number of pit stops (laps)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space={"low": 1, "high": race_length},
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=30,
        stop_criteria="saturate"
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_pit_laps = sorted([int(lap) for lap in solution])

    st.success(f"Best Pit Stop Laps: {best_pit_laps}")
    return best_pit_laps

# ============================ FETCH FASTF1 TELEMETRY ============================
def load_telemetry(season=2023, round_number=1):
    st.info("Loading real telemetry data...")
    session = fastf1.get_session(season, round_number, 'R')
    session.load()
    lap_data = session.laps.pick_driver(selected_driver.split()[-1][:3].upper())
    return lap_data

# ============================ RUNNING EVERYTHING ============================
if st.button("🏁 Run Race Simulation + Optimization"):
    best_pit_laps = run_ga()
    total_time, lap_times = simulate_race(best_pit_laps)

    laps = np.arange(1, race_length + 1)
    tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
    fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

    # === PLOTLY GRAPHS ===
    col1, col2 = st.columns(2)

    with col1:
        fig_lap_times = go.Figure()
        fig_lap_times.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', line=dict(color=team_colors[0], width=3)))
        fig_lap_times.update_layout(title='Lap Times', template='plotly_dark', height=400)
        st.plotly_chart(fig_lap_times, use_container_width=True)

    with col2:
        fig_tire_wear = go.Figure()
        fig_tire_wear.add_trace(go.Scatter(x=laps, y=tire_wear, mode='lines+markers', line=dict(color='orange', width=3)))
        fig_tire_wear.update_layout(title='Tire Wear (%)', template='plotly_dark', height=400)
        st.plotly_chart(fig_tire_wear, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_fuel_load = go.Figure()
        fig_fuel_load.add_trace(go.Scatter(x=laps, y=fuel_load, mode='lines+markers', line=dict(color='yellow', width=3)))
        fig_fuel_load.update_layout(title='Fuel Load (%)', template='plotly_dark', height=400)
        st.plotly_chart(fig_fuel_load, use_container_width=True)

    with col4:
        fig_pits = go.Figure()
        fig_pits.add_trace(go.Scatter(x=best_pit_laps, y=[pit_stop_time]*len(best_pit_laps), mode='markers', marker=dict(size=12, color='red')))
        fig_pits.update_layout(title='Pit Stops', template='plotly_dark', height=400)
        st.plotly_chart(fig_pits, use_container_width=True)

    st.success(f"Total Race Time: {total_time:.2f} seconds")

# ============================ TELEMETRY DATA PLOT ============================
if st.checkbox("Show Real Telemetry Data (FastF1)"):
    telemetry = load_telemetry(season=2023, round_number=1)
    st.write(telemetry[['LapTime', 'Compound', 'TyreLife']])
