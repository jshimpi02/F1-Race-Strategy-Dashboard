import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import pygad
import time

# === STREAMLIT CONFIG === #
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm + Live Telemetry + Multi-Driver")
st.markdown("---")

# === TEAMS & DRIVERS === #
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

# === ASSET PATHS === #
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)
st.sidebar.markdown(f"### Base Degradation Factor: {degradation_base}")

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
num_opponents = st.sidebar.slider("Number of Opponents", 2, 10, 5)

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION === #
st.sidebar.header("🚿 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

# === GENERATE SIMULATED OPPONENT DATA === #
def generate_opponents(num_opponents, race_length):
    opponents = {}
    for i in range(1, num_opponents+1):
        driver_name = f"Opponent {i}"
        lap_times = np.random.normal(90 + random.uniform(-3, 3), 1.5, size=race_length)
        opponents[driver_name] = lap_times
    return opponents

# === GENETIC ALGORITHM FITNESS FUNCTION === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = [idx for idx, decision in enumerate(solution) if decision == 1]

    total_time = 0
    tire_wear = 0

    for lap in range(race_length):
        lap_time = 90 + degradation_base * tire_wear

        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 0  # reset tire wear
        else:
            tire_wear += 1

        total_time += lap_time

    return 1.0 / total_time

# === GENETIC ALGORITHM SETUP === #
def run_genetic_algorithm():
    num_generations = 50
    num_parents_mating = 5
    sol_per_pop = 10
    num_genes = race_length

    gene_space = [0, 1]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10
    )

    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()

    pit_decisions = [idx for idx, decision in enumerate(best_solution) if decision == 1]
    return pit_decisions

# === RUN SIMULATION BUTTON === #
run_simulation = st.sidebar.button("🏎️ Run Simulation")

if run_simulation:
    with st.spinner("Running simulation with optimized pit strategy..."):
        pit_decisions = run_genetic_algorithm()

        st.sidebar.success("Simulation Complete! 🚀")
        st.markdown(f"### 🛠️ Optimized Pit Stops: {pit_decisions}")

        laps = np.arange(1, race_length + 1)
        lap_times = []
        tire_wear_data = []
        fuel_load_data = []
        tire_wear = 0
        fuel_load = 100

        for lap in laps:
            base_time = 90
            if lap in pit_decisions:
                lap_time = base_time + pit_stop_time
                tire_wear = 0
            else:
                lap_time = base_time + degradation_base * tire_wear
                tire_wear += 1

            fuel_load -= (100 / race_length)
            lap_times.append(lap_time)
            tire_wear_data.append(100 - tire_wear * degradation_base * 100)
            fuel_load_data.append(fuel_load)

        opponents = generate_opponents(num_opponents, race_length)

        telemetry_df = pd.DataFrame({"Lap": laps, selected_driver: lap_times})
        for opp_driver, opp_laps in opponents.items():
            telemetry_df[opp_driver] = opp_laps

        col1, col2 = st.columns(2)

        with col1:
            fig_times = go.Figure()
            fig_times.add_trace(go.Scatter(
                x=laps,
                y=telemetry_df[selected_driver],
                mode='lines+markers',
                name=selected_driver,
                line=dict(color=team_colors[0], width=3)
            ))

            for opp_driver in opponents.keys():
                fig_times.add_trace(go.Scatter(
                    x=laps,
                    y=telemetry_df[opp_driver],
                    mode='lines',
                    name=opp_driver,
                    line=dict(width=1)
                ))

            fig_times.update_layout(
                title="Lap Times Comparison",
                template="plotly_dark",
                xaxis_title="Lap",
                yaxis_title="Time (s)",
                height=400
            )
            st.plotly_chart(fig_times, use_container_width=True)

        with col2:
            fig_delta = go.Figure()
            deltas = telemetry_df[selected_driver] - telemetry_df.iloc[:, 2:].min(axis=1)

            fig_delta.add_trace(go.Scatter(
                x=laps,
                y=deltas,
                mode='lines+markers',
                name="Lead Delta (vs closest rival)",
                line=dict(color=team_colors[1], width=3)
            ))

            fig_delta.update_layout(
                title="Track Position Delta",
                template="plotly_dark",
                xaxis_title="Lap",
                yaxis_title="Delta Time (s)",
                height=400
            )
            st.plotly_chart(fig_delta, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_tire = go.Figure()
            fig_tire.add_trace(go.Scatter(
                x=laps,
                y=tire_wear_data,
                mode='lines+markers',
                name="Tire Wear (%)",
                line=dict(color='orange', width=3)
            ))
            fig_tire.update_layout(
                title="Tire Wear Over Race",
                template="plotly_dark",
                xaxis_title="Lap",
                yaxis_title="Tire Wear (%)",
                height=400
            )
            st.plotly_chart(fig_tire, use_container_width=True)

        with col4:
            fig_fuel = go.Figure()
            fig_fuel.add_trace(go.Scatter(
                x=laps,
                y=fuel_load_data,
                mode='lines+markers',
                name="Fuel Load (%)",
                line=dict(color='yellow', width=3)
            ))
            fig_fuel.update_layout(
                title="Fuel Load Over Race",
                template="plotly_dark",
                xaxis_title="Lap",
                yaxis_title="Fuel Load (%)",
                height=400
            )
            st.plotly_chart(fig_fuel, use_container_width=True)

        st.subheader("🔧 Pit Stop Strategy")
        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(
            x=pit_decisions,
            y=[pit_stop_time for _ in pit_decisions],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Pit Stops'
        ))
        fig_pit.update_layout(
            template='plotly_dark',
            xaxis_title='Lap',
            yaxis_title='Pit Stop Time (s)',
            height=400
        )
        st.plotly_chart(fig_pit, use_container_width=True)
