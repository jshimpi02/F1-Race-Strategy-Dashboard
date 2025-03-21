import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
from datetime import datetime
import pygad

# === CONFIGURATION === #
st.set_page_config(page_title="🏎️ F1 Race Strategy - Silverstone Live", layout="wide")
st.title("🏎️ F1 Race Strategy Dashboard - Silverstone GP 2025")

# === CACHE FOR FASTF1 === #
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# === THEME TOGGLE === #
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])
plotly_theme = "plotly_dark" if theme == "Dark" else "plotly_white"

# === TEAM SETUP === #
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "degradation_factor": 0.20, "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "degradation_factor": 0.15, "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "degradation_factor": 0.25, "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "degradation_factor": 0.30, "color": ["#FF8700", "#FFFFFF"]}
}

# === USER SELECTIONS === #
selected_team = st.sidebar.selectbox("Select Team", list(teams.keys()))

team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)

selected_driver = st.sidebar.selectbox("Select Driver", teams[selected_team]["drivers"])

driver_photo_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"
st.sidebar.image(driver_photo_path, caption=selected_driver, use_container_width=True)

degradation_base = teams[selected_team]["degradation_factor"]
team_colors = teams[selected_team]["color"]

# === SIMULATION SETTINGS === #
race_length = st.sidebar.slider("Race Length (Laps)", 30, 70, 52)
pit_stop_time = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15, 30, 22)

# === CIRCUIT + SESSION === #
year = 2024
gp_name = "Silverstone"
session_type = "R"

with st.spinner(f"Loading {gp_name} GP {session_type} session..."):
    session = fastf1.get_session(year, gp_name, session_type)
    session.load()  # historical data
    st.success(f"Loaded {gp_name} {session_type} Session!")

#circuit_info = session.get_circuit_info()
#st.subheader(f"📍 Circuit: {circuit_info.name}")
#st.write(circuit_info)

# Fetch circuit info
circuit_info = session.get_circuit_info()

# Safely get the circuit name, location, country
circuit_name = circuit_info.get('Name', 'Unknown Circuit')
location = circuit_info.get('Location', 'Unknown Location')
country = circuit_info.get('Country', 'Unknown Country')

# Display circuit details
st.subheader(f"📍 Circuit: {circuit_name}")
st.caption(f"📍 Location: {location}, {country}")

# === TELEMETRY DATA === #
drivers = session.laps["Driver"].unique()
selected_drivers = st.sidebar.multiselect("Select Drivers for Telemetry", drivers.tolist(), default=drivers[:2])

driver_telemetry = {}
for driver in selected_drivers:
    laps = session.laps.pick_driver(driver).pick_fastest()
    if laps.empty:
        continue
    telemetry = laps.get_car_data().add_distance()
    driver_telemetry[driver] = {
        "X": telemetry['X'].values,
        "Y": telemetry['Y'].values,
        "Speed": telemetry['Speed'].values,
        "DRS": telemetry['DRS'].values
    }

# === TELEMETRY ANIMATION === #
st.subheader(f"📈 Circuit Telemetry Animation - {gp_name}")

frames = []
max_frames = max([len(data["X"]) for data in driver_telemetry.values()])

for i in range(0, max_frames, 10):
    frame_data = []
    for driver, data in driver_telemetry.items():
        if len(data["X"]) > i:
            drs_color = 'green' if data["DRS"][i] == 1 else 'red'
            frame_data.append(go.Scatter(
                x=[data["X"][i]],
                y=[data["Y"][i]],
                mode='markers+text',
                marker=dict(size=10, color=drs_color),
                text=[driver],
                textposition='top center',
                name=driver
            ))
    frames.append(go.Frame(data=frame_data, name=str(i)))

layout = go.Layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    template=plotly_theme,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
        ]
    )]
)

fig_track = go.Figure(frames=frames, layout=layout)
st.plotly_chart(fig_track, use_container_width=True)

# === GENETIC ALGORITHM FOR PIT STOP === #
def fitness_func(ga_instance, solution, solution_idx):
    pit_laps = [int(round(pit)) for pit in solution if pit >= 0 and pit < race_length]
    total_time = 0
    tire_wear = 0
    current_lap = 0

    for lap in range(race_length):
        wear_penalty = degradation_base * tire_wear
        lap_time = 90 + wear_penalty
        total_time += lap_time
        tire_wear += 1

        if lap in pit_laps:
            total_time += pit_stop_time
            tire_wear = 0

    return -total_time

def run_ga():
    gene_space = [{'low': 0, 'high': race_length} for _ in range(3)]
    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=3,
        gene_space=gene_space,
        parent_selection_type="rank",
        mutation_type="random"
    )
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    pit_laps = sorted([int(round(pit)) for pit in solution if 0 <= pit < race_length])
    return pit_laps

if st.sidebar.button("🚀 Run Pit Stop Strategy"):
    with st.spinner("Running Genetic Algorithm for Pit Strategy..."):
        best_pit_stops = run_ga()
        st.sidebar.success(f"Best Pit Stops: {best_pit_stops}")
else:
    best_pit_stops = []

# === SIMULATED RACE DATA === #
laps = np.arange(1, race_length + 1)
lap_times = 90 + degradation_base * laps + np.random.normal(0, 0.5, race_length)

for pit in best_pit_stops:
    lap_times[pit:] += pit_stop_time

lead_delta = np.cumsum(np.random.normal(0, 0.5, race_length))
tire_wear = np.maximum(0, 100 - degradation_base * laps * 100)
fuel_load = np.maximum(0, 100 - (laps * (100 / race_length)))

# === LAP TIME GRAPH === #
col1, col2 = st.columns(2)

with col1:
    fig_lap_times = go.Figure()
    fig_lap_times.add_trace(go.Scatter(
        x=laps,
        y=lap_times,
        mode='lines+markers',
        line=dict(color=team_colors[0], width=3),
        name='Lap Times'
    ))
    fig_lap_times.update_layout(
        title='Lap Times Over Race',
        template=plotly_theme,
        xaxis_title='Lap',
        yaxis_title='Time (s)',
        height=400
    )
    st.plotly_chart(fig_lap_times, use_container_width=True)

with col2:
    fig_lead_delta = go.Figure()
    fig_lead_delta.add_trace(go.Scatter(
        x=laps,
        y=lead_delta,
        mode='lines+markers',
        line=dict(color=team_colors[1], width=3),
        name='Lead Delta'
    ))
    fig_lead_delta.update_layout(
        title='Lead Delta Over Race',
        template=plotly_theme,
        xaxis_title='Lap',
        yaxis_title='Delta (s)',
        height=400
    )
    st.plotly_chart(fig_lead_delta, use_container_width=True)

# === TIRE WEAR & FUEL LOAD === #
col3, col4 = st.columns(2)

with col3:
    fig_tire = go.Figure()
    fig_tire.add_trace(go.Scatter(
        x=laps,
        y=tire_wear,
        mode='lines+markers',
        line=dict(color='orange', width=3),
        name='Tire Wear'
    ))
    fig_tire.update_layout(
        title='Tire Wear Over Race',
        template=plotly_theme,
        xaxis_title='Lap',
        yaxis_title='Wear (%)',
        height=400
    )
    st.plotly_chart(fig_tire, use_container_width=True)

with col4:
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(
        x=laps,
        y=fuel_load,
        mode='lines+markers',
        line=dict(color='yellow', width=3),
        name='Fuel Load'
    ))
    fig_fuel.update_layout(
        title='Fuel Load Over Race',
        template=plotly_theme,
        xaxis_title='Lap',
        yaxis_title='Fuel (%)',
        height=400
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

# === PIT STOP STRATEGY === #
st.subheader("🔧 Pit Stop Strategy")
st.markdown(f"Pit Stops at Laps: {best_pit_stops}")

fig_pit = go.Figure()
fig_pit.add_trace(go.Scatter(
    x=best_pit_stops,
    y=[pit_stop_time] * len(best_pit_stops),
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))
fig_pit.update_layout(
    title="Pit Stop Strategy",
    template=plotly_theme,
    xaxis_title='Lap',
    yaxis_title='Pit Stop Time (s)',
    height=400
)
st.plotly_chart(fig_pit, use_container_width=True)

# === DRIVER LEADERBOARD === #
st.subheader("🏆 Driver Leaderboard")

driver_leaderboard = session.laps.groupby("Driver")["LapTime"].min().reset_index().sort_values("LapTime")
driver_leaderboard["Position"] = range(1, len(driver_leaderboard) + 1)
st.dataframe(driver_leaderboard)

# === CONSTRUCTOR STANDINGS === #
st.subheader("🏁 Constructor Standings")

constructor_data = pd.DataFrame({
    'Constructor': ["Red Bull", "Mercedes", "Ferrari", "McLaren"],
    'Points': [640, 525, 430, 389]
}).sort_values(by='Points', ascending=False)

st.bar_chart(constructor_data.set_index('Constructor'))

# === FOOTER === #
st.markdown("---")
st.caption(f"© {datetime.now().year} F1 Race Strategy | Powered by Streamlit + FastF1 + Plotly")
