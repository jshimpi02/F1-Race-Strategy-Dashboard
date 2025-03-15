import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import pygad
import fastf1
from PIL import Image

# Enable FastF1 Cache
if not os.path.exists('./cache'):
    os.makedirs('./cache')
fastf1.Cache.enable_cache('./cache')

# === PAGE CONFIG === #
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm + Multi-driver + Live Telemetry")
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
num_opponents = st.sidebar.slider("Number of Opponents", 1, 10, 5)

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
    pit_laps = sorted(set(int(lap) for lap in solution if 0 < lap <= race_length))
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

    return -total_time

def run_ga():
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=3,
        init_range_low=1,
        init_range_high=race_length,
        mutation_percent_genes=30
    )
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_pit_laps = sorted(set(int(lap) for lap in best_solution if lap >= 0 and lap <= race_length))
    return best_pit_laps

# === LIVE TELEMETRY === #
def load_live_telemetry(year=2023, gp='Bahrain'):
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load()
        laps = session.laps.pick_driver(selected_driver.split()[1][:3].upper())
        lap_times = laps['LapTime'].dt.total_seconds()
        return lap_times
    except Exception as e:
        st.error(f"Failed to load telemetry: {e}")
        return None

# === RUN SIMULATION BUTTON === #
if st.sidebar.button("🏁 Run Simulation"):
    with st.spinner("Running multi-driver race simulation..."):
        race_data = []
        drivers_to_simulate = [selected_driver]
        other_drivers = [d for team in teams.values() for d in team["drivers"] if d != selected_driver]
        drivers_to_simulate += random.sample(other_drivers, min(num_opponents, len(other_drivers)))

        for driver in drivers_to_simulate:
            profile = driver_profiles[driver]
            pit_laps = run_ga()

            laps = np.arange(1, race_length + 1)
            lap_times = []
            tire_wear = []
            fuel_load = []
            current_tire_wear = 0
            current_fuel = 100

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

                lap_time = 90 * (1 + current_tire_wear / 1000 + current_fuel / 1000) * (1 - profile["skill"]) * weather_penalty

                if lap in pit_laps:
                    lap_time += pit_stop_time
                    current_tire_wear = 0

                lap_times.append(lap_time)
                tire_wear.append(current_tire_wear)
                fuel_load.append(current_fuel)

            race_data.append({
                "driver": driver,
                "lap_times": lap_times,
                "tire_wear": tire_wear,
                "fuel_load": fuel_load,
                "pit_laps": pit_laps
            })

        # === VISUALIZATION === #
        st.header("🏎️ Multi-driver Race Simulation Results")

        fig = go.Figure()
        for data in race_data:
            fig.add_trace(go.Scatter(
                x=np.arange(1, race_length + 1),
                y=data["lap_times"],
                mode='lines+markers',
                name=data["driver"]
            ))

        fig.update_layout(
            title="Lap Times for All Drivers",
            template='plotly_dark',
            xaxis_title='Lap',
            yaxis_title='Lap Time (s)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # === PIT STRATEGY VISUAL === #
        st.subheader("🔧 Pit Stop Strategies")
        for data in race_data:
            st.markdown(f"**{data['driver']}** Pit Stops at: {data['pit_laps']}")

        # === TELEMETRY === #
        st.subheader("📊 Live Telemetry (Real Race Lap Times)")
        telemetry = load_live_telemetry()
        if telemetry is not None:
            fig_telemetry = go.Figure()
            fig_telemetry.add_trace(go.Scatter(
                y=telemetry,
                x=np.arange(1, len(telemetry) + 1),
                mode='lines+markers',
                name=f"{selected_driver} Real Lap Times"
            ))
            fig_telemetry.update_layout(
                title="Real Lap Times from FastF1",
                template='plotly_dark',
                xaxis_title='Lap',
                yaxis_title='Lap Time (s)'
            )
            st.plotly_chart(fig_telemetry, use_container_width=True)

        # === DOWNLOAD DATA === #
        st.subheader("📥 Download Race Data")
        df = pd.DataFrame({
            "Lap": np.arange(1, race_length + 1)
        })
        for data in race_data:
            df[data["driver"] + "_LapTime"] = data["lap_times"]

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "race_data.csv", "text/csv")

        st.sidebar.success("✅ Simulation Complete!")
