import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# === PHASE 7: F1 Race Strategy Simulator ===
st.set_page_config(page_title="🏎️ F1 Race Strategy RL Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - RL + Dynamic Weather + Incidents")
st.markdown("---")

# === TEAM & DRIVER SELECTION ===
st.sidebar.header("🏁 Team & Driver Selection")
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30},
}
selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]
st.sidebar.markdown(f"### Base Degradation Factor: {degradation_base}")

# === SIMULATION SETTINGS ===
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = 5

# === WEATHER SETTINGS ===
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION ===
st.sidebar.header("🛞 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

# === RL PLACEHOLDER AGENT ===
def rl_strategy(current_lap, grip_level, weather, current_tire):
    """
    Simplified RL strategy for tire decisions.
    """
    if weather == "Heavy Rain":
        return "Wets"
    if weather == "Light Rain":
        return "Intermediates"
    if current_lap % 20 == 0 and current_tire != "Soft":
        return "Soft"
    return None

# === RUN SIMULATION ===
if st.sidebar.button("Run Race Simulation 🚀"):
    st.subheader(f"Race Simulation: {selected_driver} for {selected_team}")
    st.markdown("---")

    # === Initial Degradation ===
    player_tire = selected_tire
    player_total_time = 0
    player_lap_times = []
    pit_laps = []

    tire_degradation = {
        "Soft": 0.40,
        "Medium": 0.25,
        "Hard": 0.15,
        "Intermediates": 0.30,
        "Wets": 0.35
    }
    final_degradation = degradation_base + tire_degradation[player_tire]

    # === Opponent Initialization ===
    opponents = []
    for i in range(num_opponents):
        opponent_tire = random.choice(list(tire_degradation.keys()))
        opponents.append({
            "name": f"Opponent {i+1}",
            "degradation_rate": degradation_base + tire_degradation[opponent_tire],
            "lap_times": [],
            "pit_laps": [],
            "tire": opponent_tire,
            "total_time": 0,
            "current_lap_time": 100,
            "status": "Running"
        })

    leaderboard = []
    dynamic_weather = []
    safety_car_laps = []

    # === Race Loop ===
    for lap in range(1, race_length + 1):

        # === Dynamic Weather ===
        if selected_weather == "Dynamic Weather":
            if lap == 1 or lap % 15 == 0:
                current_weather = np.random.choice(["Clear", "Light Rain", "Heavy Rain"], p=[0.5, 0.3, 0.2])
            dynamic_weather.append(current_weather)
        else:
            current_weather = selected_weather
            dynamic_weather.append(current_weather)

        grip_level = {"Clear": 1.0, "Light Rain": 0.8, "Heavy Rain": 0.6}[current_weather]
        grip_penalty = (1 - grip_level) * 20

        # === Safety Car Logic ===
        if lap % 20 == 0:
            safety_car_laps.append(lap)
            grip_penalty += 10

        # === Random Incidents ===
        for opponent in opponents:
            if opponent["status"] == "Running" and random.random() < 0.01:
                opponent["status"] = "Retired"

        # === Player RL Tire Decision ===
        new_tire = rl_strategy(lap, grip_level, current_weather, player_tire)
        if new_tire and new_tire != player_tire:
            player_tire = new_tire
            pit_laps.append(lap)
            final_degradation = degradation_base + tire_degradation[player_tire]
            lap_time = 100 + pit_stop_time + grip_penalty
        else:
            lap_time = 100 + (lap * final_degradation) + grip_penalty

        player_total_time += lap_time
        player_lap_times.append(lap_time)

        # === Opponents ===
        for opponent in opponents:
            if opponent["status"] == "Retired":
                continue

            if lap % random.choice([15, 18, 22]) == 0:
                new_tire = random.choice(list(tire_degradation.keys()))
                opponent["tire"] = new_tire
                opponent["degradation_rate"] = degradation_base + tire_degradation[new_tire]
                opponent["pit_laps"].append(lap)
                lap_time = 100 + pit_stop_time + grip_penalty
            else:
                lap_time = 100 + (lap * opponent["degradation_rate"]) + grip_penalty

            opponent["total_time"] += lap_time
            opponent["lap_times"].append(lap_time)

        # === Leaderboard ===
        drivers = [{"name": selected_driver, "total_time": player_total_time}] + [
            {"name": o["name"], "total_time": o["total_time"]} for o in opponents if o["status"] == "Running"
        ]
        sorted_drivers = sorted(drivers, key=lambda x: x["total_time"])
        leaderboard.append([d["name"] for d in sorted_drivers])

    # === Visualizations ===
    st.write("### Lap Times (Player vs Opponents)")
    laps = np.arange(1, race_length + 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(laps, player_lap_times, label=f'{selected_driver} ({player_tire})', linewidth=2)

    for opponent in opponents:
        if opponent["status"] == "Running":
            ax.plot(laps, opponent["lap_times"], label=f'{opponent["name"]} ({opponent["tire"]})', linestyle='--')

    ax.scatter(pit_laps, [player_lap_times[i - 1] for i in pit_laps], color='red', label='Your Pit Stops', zorder=5)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Times with Pit Stops, Tires & Weather Effects")
    ax.legend()
    st.pyplot(fig)

    st.write("### Leaderboard (Lap by Lap)")
    leaderboard_df = pd.DataFrame(leaderboard, columns=[f'P{i+1}' for i in range(len(leaderboard[0]))])
    leaderboard_df.index.name = 'Lap'
    st.dataframe(leaderboard_df)

    st.write("### Safety Car Laps")
    st.write(safety_car_laps if safety_car_laps else "None")

    if selected_weather == "Dynamic Weather":
        st.write("### Dynamic Weather Conditions")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(range(1, race_length + 1), [1.0 if w == "Clear" else 0.8 if w == "Light Rain" else 0.6 for w in dynamic_weather], marker='o')
        ax2.set_xlabel("Lap")
        ax2.set_ylabel("Grip Level")
        ax2.set_title("Grip Level Over Laps")
        st.pyplot(fig2)

    st.success("Race Complete! Ready for next simulation 🏁")
