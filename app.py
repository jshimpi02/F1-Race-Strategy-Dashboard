import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import pygad
from PIL import Image

# === Streamlit Settings === #
st.set_page_config(page_title="🏎️ F1 Race Strategy Simulator", layout="wide", initial_sidebar_state="expanded")

# === Sidebar === #
with st.sidebar:
    st.image("assets/f1_logo.png", width=150)
    st.title("F1 Race Strategy Simulator 2025")

    # Theme Toggle
    theme = st.radio("Choose Theme", ("Dark", "Light"))
    st.session_state["theme"] = theme.lower()

# === Circuit Background === #
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Silverstone_Circuit_2011.svg/1920px-Silverstone_Circuit_2011.svg.png');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Team & Driver Setup === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]},
}

st.header("🏎️ Select Your Team & Driver")

col_team, col_driver = st.columns(2)

with col_team:
    selected_team = st.selectbox("Select Your Team", list(teams.keys()))
    team_colors = teams[selected_team]["color"]
    st.image(f"assets/logos/{selected_team.lower().replace(' ', '_')}.png", width=120)

with col_driver:
    selected_driver = st.selectbox("Select Your Driver", teams[selected_team]["drivers"])
    st.image(f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png", width=120)

# === Leaderboards === #
st.subheader("🏆 Leaderboards")

drivers_data = {
    "Driver": ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris", "Sergio Perez"],
    "Team": ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Red Bull Racing"],
    "Points": [275, 230, 210, 190, 180],
}

df_leaderboard = pd.DataFrame(drivers_data)
st.table(df_leaderboard)

constructors_data = {
    "Constructor": ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren"],
    "Points": [455, 380, 365, 340],
}

df_constructor = pd.DataFrame(constructors_data)
st.table(df_constructor)

# === Simulation Settings === #
st.sidebar.header("Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === Genetic Algorithm Pit Strategy === #
def run_ga():
    num_pit_stops = 3

    def fitness_func(ga_instance, solution, solution_idx):
        spacing_penalty = sum([
            abs(solution[i] - solution[i - 1]) < 5
            for i in range(1, len(solution))
        ])
        return 100 - spacing_penalty

    gene_space = [{'low': 1, 'high': race_length} for _ in range(num_pit_stops)]

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_pit_stops,
        gene_space=gene_space,
        mutation_percent_genes=20,
        stop_criteria=["reach_100"]
    )

    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    return sorted([int(lap) for lap in best_solution])

best_pit_stops = run_ga()

# === Generate Race Data === #
def generate_race_data():
    laps = np.arange(1, race_length + 1)
    lap_times = np.random.normal(90, 2, size=race_length)

    # Tire wear and fuel based on team degradation and race progress
    degradation_factor = 0.25
    tire_wear = np.maximum(0, 100 - degradation_factor * laps * 100)
    fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

    # Position delta (simplified)
    lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))

    return laps, lap_times, lead_delta, tire_wear, fuel_load

laps, lap_times, lead_delta, tire_wear, fuel_load = generate_race_data()

# === Plotly Graphs === #
st.subheader("📊 Race Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_lap_times = go.Figure()
    fig_lap_times.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', name="Lap Times", line=dict(color=team_colors[0])))
    fig_lap_times.update_layout(title="Lap Times Over Race", template="plotly_dark" if theme == "Dark" else "plotly_white")
    st.plotly_chart(fig_lap_times, use_container_width=True)

with col2:
    fig_lead_delta = go.Figure()
    fig_lead_delta.add_trace(go.Scatter(x=laps, y=lead_delta, mode='lines+markers', name="Lead Delta", line=dict(color=team_colors[1])))
    fig_lead_delta.update_layout(title="Track Position Delta", template="plotly_dark" if theme == "Dark" else "plotly_white")
    st.plotly_chart(fig_lead_delta, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig_tire_wear = go.Figure()
    fig_tire_wear.add_trace(go.Scatter(x=laps, y=tire_wear, mode='lines+markers', name="Tire Wear", line=dict(color="orange")))
    fig_tire_wear.update_layout(title="Tire Wear Over Race", template="plotly_dark" if theme == "Dark" else "plotly_white")
    st.plotly_chart(fig_tire_wear, use_container_width=True)

with col4:
    fig_fuel_load = go.Figure()
    fig_fuel_load.add_trace(go.Scatter(x=laps, y=fuel_load, mode='lines+markers', name="Fuel Load", line=dict(color="yellow")))
    fig_fuel_load.update_layout(title="Fuel Load Over Race", template="plotly_dark" if theme == "Dark" else "plotly_white")
    st.plotly_chart(fig_fuel_load, use_container_width=True)

# === Pit Stop Visualization === #
st.subheader("🔧 Pit Stop Strategy")
st.write(f"Best Pit Stops (Laps): {best_pit_stops}")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=best_pit_stops,
    y=[pit_stop_time for _ in best_pit_stops],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))
fig_pit.update_layout(title="Pit Stop Laps", template="plotly_dark" if theme == "Dark" else "plotly_white")
st.plotly_chart(fig_pit, use_container_width=True)

# === Footer === #
st.markdown("---")
st.caption("F1 Race Strategy Simulator • Powered by Genetic Algorithms & Streamlit")

