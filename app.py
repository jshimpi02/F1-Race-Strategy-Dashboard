import os
import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import fastf1
import pygad

# === Enable FastF1 Cache === #
CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# === F1 Race Strategy Simulator === #
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy - Genetic Algorithm & Live Data")
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
team_colors = teams[selected_team]["color"]

# === Team Logo & Driver Photo === #
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === DRIVER PROFILES === #
driver_profiles = {
    "Lewis Hamilton": {"skill": 0.95, "aggression": 0.4, "wet_skill": 0.9},
    "Max Verstappen": {"skill": 0.97, "aggression": 0.5, "wet_skill": 0.85},
    "Charles Leclerc": {"skill": 0.93, "aggression": 0.6, "wet_skill": 0.8},
    "Lando Norris": {"skill": 0.89, "aggression": 0.45, "wet_skill": 0.82},
}
profile = driver_profiles[selected_driver]

# === FITNESS FUNCTION FOR GENETIC ALGORITHM === #
def fitness_func(ga_instance, solution, solution_idx):
    total_time = 0
    current_degradation = 1.0
    for lap in range(race_length):
        pit = solution[lap]  # 1 if pitting, 0 if not
        if pit:
            current_degradation = 1.0  # Reset tire degradation after pit
            total_time += pit_stop_time
        else:
            current_degradation -= degradation_base  # Tires degrade
        lap_time = 90 + (current_degradation * 5)
        total_time += lap_time
    return -total_time  # Minimize lap time

# === RUN GENETIC ALGORITHM === #
def run_ga():
    ga_instance = pygad.GA(
        num_generations=100,
        sol_per_pop=20,
        num_parents_mating=5,
        fitness_func=fitness_func,
        num_genes=race_length,
        gene_type=int,
        gene_space={"low": 0, "high": 1},
        mutation_probability=0.1,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        stop_criteria=["saturate_20"]  # ✅ Corrected stop_criteria
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_pit_laps = np.where(np.array(solution) == 1)[0]
    return best_pit_laps

# === RUN SIMULATION === #
run_simulation = st.sidebar.button("🏁 Run Simulation")

if run_simulation:
    with st.spinner("Running simulation..."):
        best_pit_laps = run_ga()
        
        # Generate race data
        laps = np.arange(1, race_length + 1)
        lap_times = np.random.normal(90, 2, size=race_length)
        tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
        fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

        # === VISUALS === #
        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(
                x=laps, y=lap_times, mode='lines+markers',
                name='Lap Times', line=dict(color=team_colors[0], width=3)
            ))
            fig_lap_times.update_layout(
                title='Lap Times Over Race', template='plotly_dark',
                font=dict(color='white'), xaxis_title='Lap', yaxis_title='Time (s)', height=400
            )
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(
                x=laps, y=tire_wear, mode='lines+markers',
                name='Tire Wear (%)', line=dict(color='orange', width=3)
            ))
            fig_tire_wear.update_layout(
                title='Tire Wear Over Race', template='plotly_dark',
                font=dict(color='white'), xaxis_title='Lap', yaxis_title='Tire Wear (%)', height=400
            )
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_fuel_load = go.Figure()
            fig_fuel_load.add_trace(go.Scatter(
                x=laps, y=fuel_load, mode='lines+markers',
                name='Fuel Load (%)', line=dict(color='yellow', width=3)
            ))
            fig_fuel_load.update_layout(
                title='Fuel Load Over Race', template='plotly_dark',
                font=dict(color='white'), xaxis_title='Lap', yaxis_title='Fuel Load (%)', height=400
            )
            st.plotly_chart(fig_fuel_load, use_container_width=True)

        # === PIT STRATEGY VISUAL === #
        st.subheader("🔧 Pit Stop Strategy")
        st.markdown(f"Pit Stops at Laps: {best_pit_laps}")

        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(
            x=best_pit_laps, y=[pit_stop_time for _ in best_pit_laps],
            mode='markers', marker=dict(size=12, color='red'), name='Pit Stops'
        ))
        fig_pit.update_layout(
            template='plotly_dark', font=dict(color='white'),
            xaxis_title='Lap', yaxis_title='Pit Stop Time (s)', height=400
        )
        st.plotly_chart(fig_pit, use_container_width=True)

        st.sidebar.success("Simulation Complete!")
