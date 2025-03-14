import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygad

# === F1 Race Strategy Simulator === #
st.set_page_config(page_title="🏎️ F1 Race Strategy GA Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm + Dynamic Weather + Incidents")
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
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
team_colors = teams[selected_team]["color"]
st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.markdown(f"### Base Degradation Factor: {degradation_base}")

# === DRIVER PHOTO ===
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time_loss = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
base_lap_time = st.sidebar.slider("Base Lap Time (seconds)", 85, 100, 90)
degradation_per_lap = st.sidebar.slider("Degradation Per Lap (seconds)", 0.1, 0.5, degradation_base)

# === GA FITNESS FUNCTION === #
def fitness_func(ga_instance, solution, solution_idx):
    total_time = 0
    tire_wear = 0

    for lap in range(race_length):
        lap_time = base_lap_time + (degradation_per_lap * tire_wear)
        total_time += lap_time

        if solution[lap] > 0.5:
            total_time += pit_stop_time_loss
            tire_wear = 0
        else:
            tire_wear += 1

    fitness = 1.0 / total_time
    return fitness

# === GA PARAMETERS === #
st.sidebar.header("🧬 Genetic Algorithm Settings")
num_generations = st.sidebar.slider("Generations", 50, 300, 100)
pop_size = st.sidebar.slider("Population Size", 10, 50, 20)
num_parents_mating = st.sidebar.slider("Parents Mating", 5, pop_size, 10)

# === TRAIN GA === #
train_ga = st.sidebar.button("🚀 Run Genetic Algorithm")

if train_ga:
    with st.spinner("Running Genetic Algorithm..."):

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=pop_size,
            num_genes=race_length,
            init_range_low=0,
            init_range_high=1,
            mutation_probability=0.1,
            gene_type=int
        )

        ga_instance.run()
        ga_instance.plot_fitness(title="GA - Fitness over Generations")

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        pit_decisions = [i+1 for i, val in enumerate(solution) if val == 1]

        st.sidebar.success("GA Optimization Complete!")
        st.sidebar.markdown(f"Best Fitness: **{solution_fitness:.5f}**")
        st.sidebar.markdown(f"Total Race Time: **{1/solution_fitness:.2f} sec**")

        # === RACE DATA === #
        def generate_race_data():
            laps = np.arange(1, race_length + 1)
            lap_times = []
            tire_wear = 0
            for lap in laps:
                lap_time = base_lap_time + (degradation_per_lap * tire_wear)
                if lap in pit_decisions:
                    lap_time += pit_stop_time_loss
                    tire_wear = 0
                else:
                    tire_wear += 1
                lap_times.append(lap_time)

            lap_times = np.array(lap_times)
            lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))
            tire_wear_values = np.maximum(0, 100 - degradation_per_lap * laps * 100)
            fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))
            return laps, lap_times, lead_delta, tire_wear_values, fuel_load

        laps, lap_times, lead_delta, tire_wear_values, fuel_load = generate_race_data()

        # === VISUALS === #
        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', name='Lap Times', line=dict(color=team_colors[0], width=3)))
            fig_lap_times.update_layout(title='Lap Times Over Race', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title='Lap', yaxis_title='Time (s)', height=400)
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_position_delta = go.Figure()
            fig_position_delta.add_trace(go.Scatter(x=laps, y=lead_delta, mode='lines+markers', name='Position Delta', line=dict(color=team_colors[1], width=3)))
            fig_position_delta.update_layout(title='Track Position Delta Over Race', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title='Lap', yaxis_title='Time Delta (s)', height=400)
            st.plotly_chart(fig_position_delta, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(x=laps, y=tire_wear_values, mode='lines+markers', name='Tire Wear (%)', line=dict(color='orange', width=3)))
            fig_tire_wear.update_layout(title='Tire Wear Over Race', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title='Lap', yaxis_title='Tire Wear (%)', height=400)
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        with col4:
            fig_fuel_load = go.Figure()
            fig_fuel_load.add_trace(go.Scatter(x=laps, y=fuel_load, mode='lines+markers', name='Fuel Load (%)', line=dict(color='yellow', width=3)))
            fig_fuel_load.update_layout(title='Fuel Load Over Race', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title='Lap', yaxis_title='Fuel Load (%)', height=400)
            st.plotly_chart(fig_fuel_load, use_container_width=True)

        # === PIT STRATEGY VISUAL === #
        st.subheader("🔧 Pit Stop Strategy")
        st.markdown(f"Pit Stops at Laps: {pit_decisions}")

        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(x=pit_decisions, y=[pit_stop_time_loss for _ in pit_decisions], mode='markers', marker=dict(size=12, color='red'), name='Pit Stops'))
        fig_pit.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis_title='Lap', yaxis_title='Pit Stop Time (s)', height=400)
        st.plotly_chart(fig_pit, use_container_width=True)

        st.sidebar.success("Simulation Complete!")
