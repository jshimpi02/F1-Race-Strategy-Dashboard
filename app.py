import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import pygad

# === F1 Race Strategy Simulator === #
st.set_page_config(page_title="🏎️ F1 GA Race Strategy Dashboard", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm Optimizer")
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

# === DRIVER PHOTO ===
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)
weather_penalty = st.sidebar.slider("Weather Penalty per Lap (seconds)", 0, 5, 2)
num_pit_stops = st.sidebar.slider("Max Pit Stops", 1, 3, 2)

# === Genetic Algorithm Parameters === #
st.sidebar.header("🧬 Genetic Algorithm Settings")
population_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 10, 100, 50)
mutation_probability = st.sidebar.slider("Mutation Probability", 0.01, 0.5, 0.1)

# === FITNESS FUNCTION === #
def fitness_func(solution, solution_idx):
    pit_laps = sorted([int(round(gene)) for gene in solution])
    
    total_time = 0
    tire_wear = 0
    
    for lap in range(1, race_length + 1):
        # Base lap time + degradation
        lap_time = 90 + degradation_base * tire_wear
        
        # Add pit stop if lap matches
        if lap in pit_laps:
            lap_time += pit_stop_time
            tire_wear = 0  # reset wear after pit
        
        # Weather impact
        lap_time += weather_penalty
        
        total_time += lap_time
        tire_wear += 1
    
    return -total_time  # Maximize fitness by minimizing total time

# === GENERATE INITIAL POPULATION === #
initial_population = []
for _ in range(population_size):
    pits = sorted(random.sample(range(5, race_length - 5), num_pit_stops))
    initial_population.append(pits)

# === RUN GENETIC ALGORITHM === #
def run_genetic_algorithm():
    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=population_size//2,
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=num_pit_stops,
        initial_population=initial_population,
        mutation_probability=mutation_probability,
        mutation_type="random",
        gene_space={'low': 5, 'high': race_length - 5}
    )
    ga_instance.run()
    return ga_instance

optimize = st.sidebar.button("🚀 Optimize Pit Strategy")

if optimize:
    with st.spinner("Running Genetic Algorithm Optimization..."):
        ga_instance = run_genetic_algorithm()
        solution, solution_fitness, _ = ga_instance.best_solution()
        best_pit_laps = sorted([int(round(g)) for g in solution])
        st.sidebar.success(f"Optimization Complete! Best Strategy: {best_pit_laps}")

        # === VISUALIZATION === #
        laps = np.arange(1, race_length + 1)
        lap_times = []
        tire_wear = 0
        for lap in laps:
            lap_time = 90 + degradation_base * tire_wear
            if lap in best_pit_laps:
                lap_time += pit_stop_time
                tire_wear = 0
            lap_time += weather_penalty
            lap_times.append(lap_time)
            tire_wear += 1

        tire_wear_values = [min(degradation_base * w * 100, 100) for w in range(race_length)]
        fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(
                x=laps,
                y=lap_times,
                mode='lines+markers',
                name='Lap Times',
                line=dict(color=team_colors[0], width=3)
            ))
            fig_lap_times.update_layout(
                title=f'Lap Times for {selected_driver}',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(
                x=laps,
                y=tire_wear_values,
                mode='lines+markers',
                name='Tire Wear',
                line=dict(color='orange', width=3)
            ))
            fig_tire_wear.update_layout(
                title='Tire Wear Over Laps',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        fig_pit_strategy = go.Figure()
        fig_pit_strategy.add_trace(go.Scatter(
            x=best_pit_laps,
            y=[pit_stop_time]*len(best_pit_laps),
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

        st.success(f"🏁 Best Pit Stop Strategy: {best_pit_laps}")
        st.info(f"🏆 Total Race Time: {-solution_fitness:.2f} seconds")

# === Time Complexity Analysis === #
"""
Time Complexity of Genetic Algorithm:
- Fitness Evaluation: O(race_length) per individual
- Population Size = P
- Generations = G
Total Complexity: O(P * G * race_length)

This is scalable and avoids the instability of RL.
"""
