import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

st.title("🏎️ F1 Race Strategy Simulator - RL Pit Stop Optimizer")
st.markdown("---")

# ================================
# 🏎️ TEAM & DRIVER SELECTION
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
# 🏎️ TIRE COMPOUND SELECTION
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
# 🏎️ SIMULATION SETTINGS
# ================================
st.sidebar.header("⚙️ Simulation Settings")

race_length = st.sidebar.slider("Race Length (Laps)", min_value=30, max_value=70, value=56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss", min_value=15, max_value=30, value=22)

# FINAL degradation rate after all selections
final_degradation = degradation_base + tire_degradation

st.sidebar.markdown(f"### Final Degradation per Lap: {final_degradation}")

# ================================
# 🚀 RUN SIMULATION
# ================================
if st.sidebar.button("Run Simulation 🚀"):
    st.subheader(f"Race Simulation for {selected_driver} - {selected_team}")
    st.markdown("---")

    laps = np.arange(1, race_length + 1)
    lap_times = 100 + (laps * final_degradation)
    opponent_times = 100 + (laps * (final_degradation * 1.2))  # Opponent is slower by default

    pit_stops = [15, 40]

    # Lap Times Graph
    st.write("### Lap Times & Pit Stops")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(laps, lap_times, label=f'{selected_driver} Lap Times')
    ax.plot(laps, opponent_times, label='Opponent Lap Times', linestyle='--')
    ax.scatter(pit_stops, [lap_times[i - 1] for i in pit_stops], color='red', label='Pit Stops')
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Lap Times with Pit Stops")
    ax.legend()
    st.pyplot(fig)

    # Track Position Delta
    st.write("### Track Position Delta Over Race")
    lead_time = opponent_times - lap_times
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(laps, lead_time, label='Track Position Delta (Opponent vs Agent)')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Time Difference (seconds)")
    ax2.set_title("Track Position Delta Over Race")
    ax2.legend()
    st.pyplot(fig2)

    st.success("Simulation Complete ✅")
