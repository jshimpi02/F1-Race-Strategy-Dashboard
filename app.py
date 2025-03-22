import os
import io
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Optional RL/GA imports
import pygad
# import fastf1  # Uncomment if telemetry is fully configured

# Set page config
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# === LIGHT / DARK MODE === #
theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])
plotly_template = "plotly_dark" if theme == "Dark" else "plotly_white"
bg_color = "black" if theme == "Dark" else "white"
font_color = "white" if theme == "Dark" else "black"

# === HEADER === #
st.markdown(f"<h1 style='text-align: center; color: {font_color};'>🏎️ F1 Race Strategy Simulator</h1>", unsafe_allow_html=True)
st.markdown("---")

# === TEAM & DRIVER DATA === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": "#00D2BE"},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": "#1E41FF"},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": "#DC0000"},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": "#FF8700"}
}

# === SIDEBAR === #
st.sidebar.header("🏎️ Team & Driver Selection")
selected_team = st.sidebar.selectbox("Select Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])

# === DISPLAY TEAM LOGO & DRIVER PHOTO === #
team_logo_path = f"assets/teams/{selected_team.lower().replace(' ', '_')}.png"
driver_photo_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

col1, col2 = st.sidebar.columns(2)
with col1:
    st.image(team_logo_path, width=100)
with col2:
    st.image(driver_photo_path, width=100)

# === CIRCUIT BACKGROUND === #
st.subheader("📍 Silverstone Circuit Layout")
circuit_bg = Image.open("assets/circuits/silverstone.png")
st.image(circuit_bg, caption="Silverstone Track", use_column_width=True)

# === CIRCUIT COORDINATES DATA === #
# Example circuit data - replace with real telemetry or load from CSV/JSON
corners_data = {
    "X": [0, 1, 2, 3, 4, 5, 6],
    "Y": [0, 2, 1, 3, 5, 4, 6]
}
corners_df = pd.DataFrame(corners_data)

# === CIRCUIT TRACK ANIMATION === #
fig_track = go.Figure()

fig_track.add_trace(go.Scatter(
    x=corners_df["X"],
    y=corners_df["Y"],
    mode='lines+markers',
    line=dict(color=teams[selected_team]["color"], width=3),
    marker=dict(size=8)
))

fig_track.update_layout(
    title='Circuit Path Animation',
    template=plotly_template,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=font_color),
    xaxis_title='X',
    yaxis_title='Y',
    height=500
)

st.plotly_chart(fig_track, use_container_width=True)

# === LEADERBOARD === #
st.subheader("🏆 Driver & Constructor Leaderboard (2025)")
leaderboard_data = {
    "Driver": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Lando Norris"],
    "Team": ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren"],
    "Points": [250, 200, 180, 160]
}

leaderboard_df = pd.DataFrame(leaderboard_data)
st.table(leaderboard_df)

# === GA FOR PIT STOP STRATEGY === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = np.where(np.array(solution) == 1)[0]
    if len(pit_laps) == 0:
        return 0
    total_time = 0
    base_lap_time = 90
    degradation = 0.3
    for lap in range(56):  # Example: 56 lap race
        pit_penalty = 20 if lap in pit_laps else 0
        degradation_penalty = degradation * lap
        lap_time = base_lap_time + degradation_penalty + pit_penalty
        total_time += lap_time
    fitness = 1 / total_time
    return fitness

def run_ga():
    gene_space = [0, 1]  # 0 = no pit, 1 = pit
    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=56,
        gene_space=gene_space
    )
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_pit_laps = np.where(np.array(best_solution) == 1)[0].tolist()
    return best_pit_laps

if st.sidebar.button("Run Pit Stop Strategy (GA)"):
    best_pit_stops = run_ga()
    st.success(f"Optimal Pit Stops at Laps: {best_pit_stops}")

    fig_pit = go.Figure()
    fig_pit.add_trace(go.Scatter(
        x=best_pit_stops,
        y=[20]*len(best_pit_stops),  # Example pit stop penalty time
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Pit Stops'
    ))
    fig_pit.update_layout(
        title='Pit Stop Strategy',
        template=plotly_template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        xaxis_title='Lap',
        yaxis_title='Pit Stop Time (s)',
        height=400
    )
    st.plotly_chart(fig_pit, use_container_width=True)

# === TELEMETRY PLACEHOLDER === #
st.subheader("📡 Live Telemetry (Sample Data)")
telemetry_data = {
    "Lap": list(range(1, 11)),
    "Speed (km/h)": [310, 320, 315, 325, 330, 328, 335, 340, 338, 345]
}
telemetry_df = pd.DataFrame(telemetry_data)

fig_speed = go.Figure()
fig_speed.add_trace(go.Scatter(
    x=telemetry_df["Lap"],
    y=telemetry_df["Speed (km/h)"],
    mode='lines+markers',
    line=dict(color=teams[selected_team]["color"], width=3)
))
fig_speed.update_layout(
    title='Speed Over Laps',
    template=plotly_template,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=font_color),
    xaxis_title='Lap',
    yaxis_title='Speed (km/h)',
    height=400
)
st.plotly_chart(fig_speed, use_container_width=True)

# === FOOTER === #
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: {font_color};'>© 2025 F1 Race Strategy Simulator</p>", unsafe_allow_html=True)
