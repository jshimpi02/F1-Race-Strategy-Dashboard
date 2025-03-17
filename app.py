import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import pygad
import time

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# === MODE TOGGLE ===
mode = st.sidebar.radio("Choose Mode", ("Dark Mode", "Light Mode"))

plotly_template = 'plotly_dark' if mode == "Dark Mode" else 'plotly_white'

if mode == "Dark Mode":
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: black; }
        </style>
    """, unsafe_allow_html=True)

# === HEADER ===
st.sidebar.image("assets/f1_logo.png", width=150)
st.title("🏁 F1 Race Strategy Simulator")
st.markdown("A Race Strategy Dashboard powered by Genetic Algorithms & Real-Time Analytics")

# === TEAM & DRIVER SETUP ===
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
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === SIMULATION SETTINGS ===
st.sidebar.header("⚙️ Simulation Settings")
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 56)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === GENETIC ALGORITHM CONFIG ===
def fitness_func(ga_instance, solution, solution_idx):
    pit_stops = solution
    time = 0
    tire_wear = 0
    fuel_load = 100

    for lap in range(race_length):
        if lap in pit_stops:
            tire_wear = 0
            time += pit_stop_time
        
        tire_wear += degradation_base
        lap_time = 90 + (tire_wear * 10) + (fuel_load * 0.05)
        time += lap_time
        fuel_load -= (100 / race_length)
    
    return -time  # Negative because PyGAD maximizes fitness

def run_ga():
    race_length = 56  # Customize if your race length changes
    num_pit_stops = 3  # You can adjust this based on strategy

    def fitness_func(ga_instance, solution, solution_idx):
        # Dummy fitness calculation for now
        # Example: penalize close pit stops, reward spacing
        spacing_penalty = sum([
            abs(solution[i] - solution[i - 1]) < 5  # penalize pit stops too close
            for i in range(1, len(solution))
        ])
        return 100 - spacing_penalty  # Higher is better

    gene_space = [{'low': 1, 'high': race_length} for _ in range(num_pit_stops)]

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_pit_stops,
        gene_space=gene_space,
        mutation_percent_genes=20,
        stop_criteria=["reach_100"]  # Example stopping criteria
    )

    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best Pit Stop Strategy (Laps): {best_solution}")
    print(f"Fitness Score: {best_solution_fitness}")

    return best_solution


# === SESSION STATE TO CONTROL RE-RUNS ===
if "race_run" not in st.session_state:
    st.session_state.race_run = False

# === ACTION BUTTONS ===
if st.sidebar.button("🏁 Run Race Simulation"):
    st.session_state.race_run = True

# === RUNNING THE SIMULATION ===
if st.session_state.race_run:
    with st.spinner("Running Race Strategy Optimization..."):
        time.sleep(1)

        best_pit_stops = run_ga()

        # Generate race data
        laps = np.arange(1, race_length + 1)
        lap_times = []
        tire_wear_list = []
        fuel_load_list = []

        tire_wear = 0
        fuel_load = 100
        total_time = 0

        for lap in laps:
            if lap in best_pit_stops:
                tire_wear = 0
                total_time += pit_stop_time

            tire_wear += degradation_base
            lap_time = 90 + (tire_wear * 10) + (fuel_load * 0.05)
            lap_times.append(lap_time)
            tire_wear_list.append(100 - tire_wear * 100)
            fuel_load_list.append(fuel_load)
            fuel_load -= (100 / race_length)

        st.success(f"Race Completed! 🏆 Best Pit Stops at Laps: {best_pit_stops}")

        # === PLOTS ===
        col1, col2 = st.columns(2)

        with col1:
            fig_lap_times = go.Figure()
            fig_lap_times.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines+markers', name='Lap Times',
                                               line=dict(color=team_colors[0], width=3)))
            fig_lap_times.update_layout(template=plotly_template, title='Lap Times Over Race', xaxis_title='Lap', yaxis_title='Time (s)')
            st.plotly_chart(fig_lap_times, use_container_width=True)

        with col2:
            fig_tire_wear = go.Figure()
            fig_tire_wear.add_trace(go.Scatter(x=laps, y=tire_wear_list, mode='lines+markers', name='Tire Wear',
                                               line=dict(color='orange', width=3)))
            fig_tire_wear.update_layout(template=plotly_template, title='Tire Wear Over Race', xaxis_title='Lap', yaxis_title='Tire Wear (%)')
            st.plotly_chart(fig_tire_wear, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_fuel = go.Figure()
            fig_fuel.add_trace(go.Scatter(x=laps, y=fuel_load_list, mode='lines+markers', name='Fuel Load',
                                          line=dict(color='yellow', width=3)))
            fig_fuel.update_layout(template=plotly_template, title='Fuel Load Over Race', xaxis_title='Lap', yaxis_title='Fuel Load (%)')
            st.plotly_chart(fig_fuel, use_container_width=True)

        with col4:
            fig_pits = go.Figure()
            fig_pits.add_trace(go.Scatter(x=best_pit_stops, y=[pit_stop_time]*len(best_pit_stops), mode='markers',
                                          marker=dict(size=12, color='red'), name='Pit Stops'))
            fig_pits.update_layout(template=plotly_template, title='Pit Stop Strategy', xaxis_title='Lap', yaxis_title='Pit Stop Time (s)')
            st.plotly_chart(fig_pits, use_container_width=True)

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Jaimin Shimpi")
