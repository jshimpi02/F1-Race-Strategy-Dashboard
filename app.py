import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Page settings
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# Title
st.title("🏎️ F1 Race Strategy Simulator - Multi-Driver + Dynamic Weather")
st.markdown("---")

# ================================
# 🏁 TEAM & DRIVER SELECTION
# ================================
st.sidebar.header("🏁 Team & Driver Selection")

teams = {
    "Mercedes": {
        "drivers": ["Lewis Hamilton", "George Russell"],
        "degradation_factor": 0.20,
    },
    "Red Bull Racing": {
        "drivers": ["Max Verstappen", "Sergio Perez"],
        "degradation_factor": 0.15,
    },
    "Ferrari": {
        "drivers": ["Charles Leclerc", "Carlos Sainz"],
        "degradation_factor": 0.25,
    },
    "McLaren": {
        "drivers": ["Lando Norris", "Oscar Piastri"],
        "degradation_factor": 0.30,
    },
}

selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
degradation_base = teams[selected_team]["degradation_factor"]

st.sidebar.markdown(f"### Degradation Factor for {selected_driver}: {degradation_base}")

# ================================
# 🛞 TIRE COMPOUND SELECTION
# ================================
st.sidebar.header("🛞 Tire Compound Selection")

tires = {
    "Soft": 0.40,
    "Medium": 0.25,
    "Hard": 0.15
}

selected_tire = st.sidebar.radio("Select Tire Compound", list(tires.keys()))
tire_degradation = tires[selected_tire]

# ================================
# ⚙️ SIMULATION SETTINGS
# ================================
st.sidebar.header("⚙️ Simulation Settings")

race_length = st.sidebar.slider("Race Length (Laps)", min_value=30, max_value=70, value=56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss", min_value=15, max_value=30, value=22)

# Final degradation for your driver
final_degradation = degradation_base + tire_degradation

# ================================
# 🌦️ WEATHER SETTINGS
# ================================
st.sidebar.header("🌦️ Weather Settings")

weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather Condition", weather_types)

# Grip level based on weather
if selected_weather == "Clear":
    grip_level = 1.0
elif selected_weather == "Light Rain":
    grip_level = 0.8
elif selected_weather == "Heavy Rain":
    grip_level = 0.6
elif selected_weather == "Dynamic Weather":
    grip_level = None  # We'll vary it lap by lap

st.sidebar.markdown(f"### Base Grip Level: {grip_level if grip_level else 'Dynamic'}")

# ================================
# 🚀 RUN SIMULATION
# ================================
if st.sidebar.button("Run Simulation 🚀"):
    st.subheader(f"Race Simulation for {selected_driver} - {selected_team}")
    st.markdown("---")

    # === Opponent Initialization ===
    num_opponents = 5
    opponents = []

    for i in range(num_opponents):
        opponents.append({
            "name": f"Opponent {i+1}",
            "degradation_rate": np.random.uniform(0.15, 0.35),
            "lap_times": [],
            "pit_laps": sorted(np.random.choice(range(10, race_length - 10), 2, replace=False)),
            "total_time": 0,
            "current_lap_time": 100
        })

    # === Player Initialization ===
    agent_total_time = 0
    agent_lap_times = []
    pit_stops = [15, 40]
    leaderboard = []
    dynamic_weather = []

    # === Run the Race ===
    for lap in range(1, race_length + 1):

        # --- Dynamic Weather Updates ---
        if selected_weather == "Dynamic Weather":
            if lap % 15 == 0:
                grip_level = np.random.choice([1.0, 0.8, 0.6], p=[0.5, 0.3, 0.2])
            dynamic_weather.append(grip_level)
        else:
            dynamic_weather.append(grip_level)

        grip_penalty = (1 - grip_level) * 20  # Grip impact on lap time

        # === Agent Lap Time ===
        if lap in pit_stops:
            pit_penalty = pit_stop_time
            agent_lap_time = 100 + pit_penalty + grip_penalty
        else:
            agent_lap_time = 100 + (lap * final_degradation) + grip_penalty

        agent_total_time += agent_lap_time
        agent_lap_times.append(agent_lap_time)

        # === Opponent Lap Times ===
        for opponent in opponents:
            if lap in opponent["pit_laps"]:
                pit_penalty = pit_stop_time
                opponent["current_lap_time"] = 100 + pit_penalty + grip_penalty
            else:
                opponent["current_lap_time"] += opponent["degradation_rate"]
                opponent["current_lap_time"] += grip_penalty

            opponent["total_time"] += opponent["current_lap_time"]
            opponent["lap_times"].append(opponent["current_lap_time"])

        # === Leaderboard Sorting ===
        drivers = [{"name": selected_driver, "total_time": agent_total_time}] + [
            {"name": o["name"], "total_time": o["total_time"]} for o in opponents
        ]

        sorted_drivers = sorted(drivers, key=lambda x: x["total_time"])
        leaderboard.append([p["name"] for p in sorted_drivers])

    # === Visualize Lap Times ===
    st.write("### Lap Times & Pit Stops")
    laps = np.arange(1, race_length + 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(laps, agent_lap_times, label=f'{selected_driver} Lap Times', linewidth=2)

    for opponent in opponents:
        ax.plot(laps, opponent["lap_times"], label=opponent["name"], linestyle='--')

    ax.scatter(pit_stops, [agent_lap_times[i - 1] for i in pit_stops], color='red', label='Pit Stops', zorder=5)

    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Times with Pit Stops & Weather Effects")
    ax.legend()
    st.pyplot(fig)

    # === Leaderboard Over Race ===
    st.write("### Leaderboard Over Race (Lap by Lap)")

    leaderboard_df = pd.DataFrame(leaderboard, columns=[f'P{i+1}' for i in range(len(opponents) + 1)])
    leaderboard_df.index.name = 'Lap'
    st.dataframe(leaderboard_df)

    # === Lead Time vs Closest Rival ===
    st.write("### Lead Time Over Closest Rival")

    lead_times = []
    for idx, lap_data in enumerate(leaderboard):
        leader = lap_data[0]
        runner_up = lap_data[1]

        leader_time = agent_total_time if leader == selected_driver else next(o["total_time"] for o in opponents if o["name"] == leader)
        runner_up_time = agent_total_time if runner_up == selected_driver else next(o["total_time"] for o in opponents if o["name"] == runner_up)

        lead_times.append(runner_up_time - leader_time)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(range(1, race_length + 1), lead_times, label='Lead Time Over Closest Rival')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Time Difference (seconds)")
    ax2.set_title("Lead Time Over Closest Rival")
    ax2.legend()
    st.pyplot(fig2)

    # === Dynamic Weather Chart ===
    if selected_weather == "Dynamic Weather":
        st.write("### Weather Conditions Over Race")

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(range(1, race_length + 1), dynamic_weather, marker='o')
        ax3.set_xlabel("Lap")
        ax3.set_ylabel("Grip Level (1 = Dry, 0.6 = Heavy Rain)")
        ax3.set_title("Track Grip Conditions Over Race")
        st.pyplot(fig3)

    st.success("Race Simulation Complete ✅")
