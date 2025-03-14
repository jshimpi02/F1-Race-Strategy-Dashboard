import os
os.environ["PYTORCH_JIT"] = "0"  # Safe default, even without RL

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygad

# === PAGE CONFIG === #
st.set_page_config(page_title="🏎️ F1 Race Strategy - Genetic Algorithm", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator - Genetic Algorithm Optimization")
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
team_colors = teams[selected_team]["color"]

team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)

driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS === #
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === WEATHER SETTINGS === #
st.sidebar.header("🌦️ Weather Settings")
weather_types = ["Clear", "Light Rain", "Heavy Rain", "Dynamic Weather"]
selected_weather = st.sidebar.selectbox("Select Weather", weather_types)

# === TIRE COMPOUND SELECTION === #
st.sidebar.header("🚾 Tire Compound Selection")
tire_options = {"Soft": 0.40, "Medium": 0.25, "Hard": 0.15}
selected_tire = st.sidebar.selectbox("Starting Tire Compound", list(tire_options.keys()))

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

# === GENETIC ALGORITHM FITNESS FUNCTION === #
def fitness_func(ga_instance, solution, solution_idx):
    total_time = 0
    tire_wear = 100
    pit_stop_penalty = pit_stop_time

    pit_laps = [lap for lap, p in enumerate(solution) if p == 1]

    if not pit_laps:
        return -999999  # Penalize no pit stops heavily

    for lap in range(race_length):
        # Add lap degradation, reduced by tire wear
        degradation = degradation_base * lap
        lap_time = 90 + degradation

        if lap in pit_laps:
            tire_wear = 100
            lap_time += pit_stop_penalty

        total_time += lap_time

    fitness = -total_time  # We minimize total time
    return fitness

# === RUN GENETIC ALGORITHM BUTTON === #
if st.sidebar.button("🚀 Optimize Pit Strategy"):
    with st.spinner("Running Genetic Algorithm..."):
        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=race_length,
            gene_type=int,
            gene_space=[0, 1],
            parent_selection_type="rank",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=5
        )

        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        pit_stop_laps = [lap for lap, pit in enumerate(best_solution) if pit == 1]

        if not pit_stop_laps:
            st.warning("⚠️ No pit stops selected by the strategy!")
        else:
            st.success(f"✅ Pit Stops at Laps: {pit_stop_laps}")

        # === RACE DATA SIMULATION === #
        laps = np.arange(1, race_length + 1)
        lap_times = np.random.normal(90, 2, size=race_length)
        lead_delta = np.cumsum(np.random.normal(0, 1, size=race_length))
        tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
        fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

        # === VISUALIZATIONS === #
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
                title='Lap Times Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Time (s)',
                height=400
            )
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_position_delta = go.Figure()
            fig_position_delta.add_trace(go.Scatter(
                x=laps,
                y=lead_delta,
                mode='lines+markers',
                name='Position Delta',
                line=dict(color=team_colors[1], width=3)
            ))
            fig_position_delta.update_layout(
                title='Track Position Delta Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Time Delta (s)',
                height=400
            )
            st.plotly_chart(fig_position_delta, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(
                x=laps,
                y=tire_wear,
                mode='lines+markers',
                name='Tire Wear (%)',
                line=dict(color='orange', width=3)
            ))
            fig_tire_wear.update_layout(
                title='Tire Wear Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Tire Wear (%)',
                height=400
            )
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        with col4:
            fig_fuel_load = go.Figure()
            fig_fuel_load.add_trace(go.Scatter(
                x=laps,
                y=fuel_load,
                mode='lines+markers',
                name='Fuel Load (%)',
                line=dict(color='yellow', width=3)
            ))
            fig_fuel_load.update_layout(
                title='Fuel Load Over Race',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Lap',
                yaxis_title='Fuel Load (%)',
                height=400
            )
            st.plotly_chart(fig_fuel_load, use_container_width=True)

        # === PIT STRATEGY VISUAL === #
        st.subheader("🔧 Pit Stop Strategy")
        fig_pit = go.Figure()

        if pit_stop_laps:
            fig_pit.add_trace(go.Scatter(
                x=pit_stop_laps,
                y=[pit_stop_time for _ in pit_stop_laps],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Pit Stops'
            ))

        fig_pit.update_layout(
            title='Pit Stop Strategy',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title='Lap',
            yaxis_title='Pit Stop Time (s)',
            height=400
        )

        st.plotly_chart(fig_pit, use_container_width=True)

        st.sidebar.success("✅ Optimization Complete!")

