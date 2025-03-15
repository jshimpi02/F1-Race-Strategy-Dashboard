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
import time
from gtts import gTTS
import base64

# === PAGE CONFIG === #
st.set_page_config(page_title="🏎️ F1 Race Strategy & Replay", layout="wide")

# === STYLE === #
st.markdown("""
<style>
.stApp {
    background-color: #0d1117;
    color: white;
}
.block-container {
    padding: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Grand Prix Dashboard + Live Commentary & Replay")

# === ENABLE CACHE === #
if not os.path.exists('./cache'):
    os.makedirs('./cache')
fastf1.Cache.enable_cache('./cache')

# === TEAM CONFIG === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

driver_profiles = {
    "Lewis Hamilton": {"skill": 0.95},
    "George Russell": {"skill": 0.90},
    "Max Verstappen": {"skill": 0.97},
    "Sergio Perez": {"skill": 0.91},
    "Charles Leclerc": {"skill": 0.93},
    "Carlos Sainz": {"skill": 0.92},
    "Lando Norris": {"skill": 0.89},
    "Oscar Piastri": {"skill": 0.88}
}

# === CIRCUIT CONFIG === #
circuits = {
    "Monaco GP": {"base_lap_time": 75, "degradation": 0.30},
    "Monza GP": {"base_lap_time": 70, "degradation": 0.20},
    "Spa GP": {"base_lap_time": 95, "degradation": 0.25},
}

# === SIDEBAR === #
st.sidebar.header("🏎️ Setup")
selected_team = st.sidebar.selectbox("Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Driver", teams[selected_team]["drivers"])
selected_circuit = st.sidebar.selectbox("Circuit", list(circuits.keys()))
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 50)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 20)

# === ASSET PATHS === #
ASSET_FOLDER = 'assets'
LOGOS_FOLDER = f"{ASSET_FOLDER}/logos"
DRIVERS_FOLDER = f"{ASSET_FOLDER}/drivers"

# === IMAGES === #
team_logo = f"{LOGOS_FOLDER}/{selected_team.lower().replace(' ', '_')}.png"
driver_photo = f"{DRIVERS_FOLDER}/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo, caption=selected_team)
st.sidebar.image(driver_photo, caption=selected_driver)

# === GLOBAL STATE === #
driver_points = {d: 0 for t in teams.values() for d in t["drivers"]}
constructor_points = {t: 0 for t in teams.keys()}

# === FITNESS FUNCTION FOR GA === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = sorted(set(int(lap) for lap in solution if 0 < lap <= race_length))
    circuit_data = circuits[selected_circuit]
    base_lap = circuit_data["base_lap_time"]
    degradation = circuit_data["degradation"]

    total_time = 0
    tire_wear = 0
    fuel_load = 100
    skill = driver_profiles[selected_driver]["skill"]

    for lap in range(1, race_length + 1):
        tire_wear += degradation * 100 / race_length
        fuel_load -= 100 / race_length
        lap_time = base_lap * (1 + tire_wear / 1000 + fuel_load / 1000) * (1 - skill)

        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 0

        total_time += lap_time

    return -total_time

def run_ga():
    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=3,
        init_range_low=1,
        init_range_high=race_length,
        mutation_percent_genes=15
    )
    ga_instance.run()
    best_solution, _, _ = ga_instance.best_solution()
    return sorted(set(int(lap) for lap in best_solution if lap > 0 and lap <= race_length))

# === AUDIO COMMENTARY === #
def generate_commentary(driver, lap, event):
    text = f"Lap {lap}: {driver} {event}"
    tts = gTTS(text=text, lang='en')
    filename = f"audio/commentary_{driver}_{lap}.mp3"
    if not os.path.exists('audio'):
        os.makedirs('audio')
    tts.save(filename)
    return filename

def play_audio(filename):
    audio_file = open(filename, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

# === SIMULATE RACE === #
if st.sidebar.button("🏁 Start Race"):
    drivers_to_simulate = [selected_driver] + random.sample(
        [d for t in teams.values() for d in t["drivers"] if d != selected_driver],
        4
    )

    race_results = []
    progress_bar = st.progress(0)

    for driver in drivers_to_simulate:
        profile = driver_profiles[driver]
        pit_laps = run_ga()

        laps = []
        lap_times = []
        tire_wear = []
        fuel_load = []
        commentary_events = []

        circuit_data = circuits[selected_circuit]
        base_lap = circuit_data["base_lap_time"]
        degradation = circuit_data["degradation"]

        tw = 0
        fuel = 100

        for lap in range(1, race_length + 1):
            tw += degradation * 100 / race_length
            fuel -= 100 / race_length
            lap_time = base_lap * (1 + tw / 1000 + fuel / 1000) * (1 - profile["skill"])

            if lap in pit_laps:
                lap_time += pit_stop_time
                tw = 0
                event = "pits!"
                commentary_events.append(generate_commentary(driver, lap, event))
            else:
                event = "is racing smoothly"
                if random.random() < 0.05:
                    event = "sets the fastest lap!"
                commentary_events.append(generate_commentary(driver, lap, event))

            laps.append(lap)
            lap_times.append(lap_time)
            tire_wear.append(tw)
            fuel_load.append(fuel)

        race_results.append({
            "driver": driver,
            "laps": laps,
            "lap_times": lap_times,
            "tire_wear": tire_wear,
            "fuel_load": fuel_load,
            "pit_laps": pit_laps,
            "total_time": sum(lap_times),
            "commentary": commentary_events
        })

    # === Sort by total time === #
    race_results.sort(key=lambda x: x["total_time"])

    # === Leaderboards === #
    points_system = [25, 18, 15, 12, 10]
    for idx, data in enumerate(race_results):
        driver_points[data["driver"]] += points_system[idx]
        for team, t in teams.items():
            if data["driver"] in t["drivers"]:
                constructor_points[team] += points_system[idx]

    # === Results === #
    st.header("🏁 Final Results")
    df_results = pd.DataFrame({
        "Position": list(range(1, len(race_results) + 1)),
        "Driver": [r["driver"] for r in race_results],
        "Total Time (s)": [round(r["total_time"], 2) for r in race_results]
    })
    st.table(df_results)

    # === Race Replay === #
    st.subheader("🎥 Race Replay & Audio Commentary")

    for lap in range(1, race_length + 1):
        st.markdown(f"### Lap {lap}")
        fig = go.Figure()

        for data in race_results:
            if lap <= len(data["laps"]):
                fig.add_trace(go.Scatter(
                    x=[lap],
                    y=[data["lap_times"][lap - 1]],
                    mode='markers',
                    marker=dict(size=12),
                    name=data["driver"]
                ))
                # Play commentary for this lap
                play_audio(data["commentary"][lap - 1])

        fig.update_layout(
            title=f"Lap {lap} Times",
            xaxis_title="Lap",
            yaxis_title="Lap Time (s)",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
        time.sleep(1)

    # === Leaderboards === #
    st.subheader("🏆 Driver Championship Leaderboard")
    st.table(pd.DataFrame(driver_points.items(), columns=["Driver", "Points"]).sort_values(by="Points", ascending=False))

    st.subheader("🏆 Constructor Championship Leaderboard")
    st.table(pd.DataFrame(constructor_points.items(), columns=["Team", "Points"]).sort_values(by="Points", ascending=False))

