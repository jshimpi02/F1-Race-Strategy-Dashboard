import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
from datetime import datetime
import time

# === F1 Race Strategy Simulator with Live Telemetry + GA + Multi-Driver === #
st.set_page_config(page_title="🏎️ F1 Race Strategy RL Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Dashboard - GA Optimized + Live Telemetry + Multi-Driver")
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
selected_drivers = st.sidebar.multiselect("Select Drivers", teams[selected_team]["drivers"], default=teams[selected_team]["drivers"])
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

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

def simulate_driver(driver):
    skill = driver_profiles[driver]["skill"]
    lap_times = []
    tire_wear = []
    fuel_load = []
    speed_data = []
    gear_data = []
    rpm_data = []
    throttle_data = []
    brake_data = []
    
    for lap in range(1, race_length + 1):
        base_time = 90
        degradation = degradation_base * lap
        weather_penalty = 5 if "Rain" in selected_weather else 0
        lap_time = base_time + degradation + random.uniform(-1, 1) * (1 - skill) + weather_penalty
        lap_times.append(lap_time)
        tire_wear.append(max(0, 100 - degradation_base * lap * 100))
        fuel_load.append(max(0, 100 - lap * (100 / race_length)))

        # Live telemetry (simulated)
        speed_data.append(random.randint(300, 360))
        gear_data.append(random.randint(1, 8))
        rpm_data.append(random.randint(11000, 15000))
        throttle_data.append(random.uniform(0.7, 1.0))
        brake_data.append(random.uniform(0.0, 0.3))
        
    return {
        "laps": list(range(1, race_length + 1)),
        "lap_times": lap_times,
        "tire_wear": tire_wear,
        "fuel_load": fuel_load,
        "speed": speed_data,
        "gear": gear_data,
        "rpm": rpm_data,
        "throttle": throttle_data,
        "brake": brake_data
    }

# === GENETIC ALGORITHM PIT STRATEGY === #
def fitness_func(solution, solution_idx):
    pit_stops = solution
    if not pit_stops:
        return 1e-6
    time_loss = len(pit_stops) * pit_stop_time
    lap_penalty = sum([degradation_base * lap for lap in pit_stops])
    total_time = race_length * 90 + time_loss + lap_penalty
    return 1 / total_time

import pygad

num_pit_stops = 2
ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=num_pit_stops,
    init_range_low=1,
    init_range_high=race_length,
    mutation_percent_genes=10
)

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
pit_decisions = sorted(list(map(int, solution)))

st.sidebar.subheader("Optimized Pit Stops")
st.sidebar.write(f"Pit stops at laps: {pit_decisions}")

# === SIMULATE & PLOT === #
st.header("📊 Race Simulation & Telemetry")

for driver in selected_drivers:
    st.subheader(f"Driver: {driver}")
    driver_data = simulate_driver(driver)
    col1, col2 = st.columns(2)

    # Lap Times
    with col1:
        fig_lap_times = go.Figure(go.Scatter(
            x=driver_data["laps"],
            y=driver_data["lap_times"],
            mode='lines+markers',
            line=dict(color=team_colors[0], width=3)
        ))
        fig_lap_times.update_layout(title="Lap Times", template="plotly_dark")
        st.plotly_chart(fig_lap_times, use_container_width=True)

    # Tire Wear
    with col2:
        fig_tire = go.Figure(go.Scatter(
            x=driver_data["laps"],
            y=driver_data["tire_wear"],
            mode='lines+markers',
            line=dict(color="orange", width=3)
        ))
        fig_tire.update_layout(title="Tire Wear", template="plotly_dark")
        st.plotly_chart(fig_tire, use_container_width=True)

    # Fuel Load
    fig_fuel = go.Figure(go.Scatter(
        x=driver_data["laps"],
        y=driver_data["fuel_load"],
        mode='lines+markers',
        line=dict(color="yellow", width=3)
    ))
    fig_fuel.update_layout(title="Fuel Load", template="plotly_dark")
    st.plotly_chart(fig_fuel, use_container_width=True)

    # === Live Telemetry === #
    st.subheader("Live Telemetry")
    telemetry_df = pd.DataFrame({
        "Lap": driver_data["laps"],
        "Speed (km/h)": driver_data["speed"],
        "Gear": driver_data["gear"],
        "RPM": driver_data["rpm"],
        "Throttle": driver_data["throttle"],
        "Brake": driver_data["brake"]
    })
    st.dataframe(telemetry_df)

# === PIT STRATEGY VISUAL === #
fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=pit_decisions,
    y=[pit_stop_time for _ in pit_decisions],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))
fig_pit.update_layout(title="Pit Stop Strategy", template="plotly_dark")
st.plotly_chart(fig_pit, use_container_width=True)

st.sidebar.success("Simulation Complete!")
