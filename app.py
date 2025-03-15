import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import pygad
from PIL import Image

# === PAGE CONFIG === #
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm + Multi-driver + Telemetry")
st.markdown("---")

# === ASSETS CONFIG === #
ASSET_FOLDER = 'assets'
LOGOS_FOLDER = f"{ASSET_FOLDER}/logos"
DRIVERS_FOLDER = f"{ASSET_FOLDER}/drivers"

# === TEAM & DRIVER SELECTION === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

st.sidebar.header("🏎️ Team & Driver Selection")
selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
team_colors = teams[selected_team]["color"]

# === LOAD LOGO AND DRIVER PHOTO === #
logo_path = f"{LOGOS_FOLDER}/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"{DRIVERS_FOLDER}/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

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

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = 5

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION === #
st.sidebar.header("🚾 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))


# === GENETIC ALGORITHM CONFIG === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = [lap for lap in solution if lap >= 0 and lap <= race_length]
    pit_laps = sorted(set(int(lap) for lap in pit_laps))
    total_time = 0
    tire_wear = 0
    fuel_load = 100
    skill = profile["skill"]

    for lap in range(1, race_length + 1):
        tire_wear += degradation_base * 100 / race_length
        fuel_load -= 100 / race_length

        weather_penalty = 1.0
        if selected_weather == "Light Rain":
            weather_penalty = 1.05
        elif selected_weather == "Heavy Rain":
            weather_penalty = 1.1
        elif selected_weather == "Dynamic Weather" and lap % 10 == 0:
            weather_penalty = 1.05 + random.choice([0.0, 0.05])

        lap_time = 90 * (1 + tire_wear / 1000 + fuel_load / 1000) * (1 - skill) * weather_penalty

        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 0

        total_time += lap_time

    return -total_time  # Minimize total time


def run_ga():
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=3,  # Number of pit stops to plan
        init_range_low=1,
        init_range_high=race_length,
        mutation_percent_genes=30
    )
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_pit_laps = sorted(set(int(lap) for lap in best_solution if lap >= 0 and lap <= race_length))
    return best_pit_laps


# === RUN SIMULATION BUTTON === #
if st.sidebar.button("🏁 Run Simulation"):
    with st.spinner("Optimizing strategy and running simulation..."):
        best_pit_laps = run_ga()

        # === GENERATE DATA FOR DRIVER === #
        laps = np.arange(1, race_length + 1)
        lap_times = []
        tire_wear = []
        fuel_load = []
        skill = profile["skill"]

        current_tire_wear = 0
        current_fuel = 100
        pit_laps = best_pit_laps

        for lap in laps:
            current_tire_wear += degradation_base * 100 / race_length
            current_fuel -= 100 / race_length

            weather_penalty = 1.0
            if selected_weather == "Light Rain":
                weather_penalty = 1.05
            elif selected_weather == "Heavy Rain":
                weather_penalty = 1.1
            elif selected_weather == "Dynamic Weather" and lap % 10 == 0:
                weather_penalty = 1.05 + random.choice([0.0, 0.05])

            lap_time = 90 * (1 + current_tire_wear / 1000 + current_fuel / 1000) * (1 - skill) * weather_penalty

            if lap in pit_laps:
                lap_time += pit_stop_time
                current_tire_wear = 0

            lap_times.append(lap_time)
            tire_wear.append(current_tire_wear)
            fuel_load.append(current_fuel)

        # === VISUALIZATION === #
        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(
                x=laps,
                y=lap_times,
                mode='lines+markers',
                name=f"{selected_driver} Lap Times",
                line=dict(color=team_colors[0], width=3)
            ))
            fig_lap_times.update_layout(
                title=f"{selected_driver} Lap Times",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
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
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
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
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_fuel_load, use_container_width=True)

        with col4:
            fig_pit_strategy = go.Figure()
            fig_pit_strategy.add_trace(go.Scatter(
                x=pit_laps,
                y=[pit_stop_time] * len(pit_laps),
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Pit Stops'
            ))
            fig_pit_strategy.update_layout(
                title='Pit Stop Strategy',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_pit_strategy, use_container_width=True)

        st.sidebar.success(f"🏁 Simulation Complete! Best Pit Stops at Laps: {pit_laps}")
