import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import time
import os
from datetime import datetime

# === Streamlit Page Setup ===
st.set_page_config(page_title="🏎️ F1 Race Strategy Dashboard", layout="wide")

# === Dark/Light Mode Toggle ===
mode = st.sidebar.radio("Choose Mode", ("Dark", "Light"))
if mode == "Dark":
    plotly_template = "plotly_dark"
    background_color = "#111111"
else:
    plotly_template = "plotly_white"
    background_color = "#FFFFFF"

# === Team and Driver Data ===
teams_2025 = {
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]}
}

# === Sidebar Selections ===
st.sidebar.image("assets/f1_logo.png", width=150)
selected_team = st.sidebar.selectbox("Select Team", list(teams_2025.keys()))
selected_driver = st.sidebar.selectbox("Select Driver", teams_2025[selected_team]["drivers"])

# === Display Team Logo and Driver Headshot ===
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# === Circuit Info ===
st.title(f"🏁 {selected_driver} | {selected_team} - Silverstone GP")
st.markdown(f"📍 **Circuit**: Silverstone International Circuit")
st.markdown("---")

# === Silverstone Circuit Data ===
rotation = 92
circuit_corners = pd.DataFrame({
    'X': [1192.51, 2770.32, 4845.32, 5802.70, 6232.29, 631.39, -566.25, 761.34, 5893.89, 7295.83],
    'Y': [4503.83, 4462.89, 5895.13, 4733.52, 6458.98, 10910.23, 9540.38, 12361.56, 12947.21, 7780.46]
})

# Normalize/Rotate Coordinates (basic adjustment)
x_coords = circuit_corners['X'] - circuit_corners['X'].min()
y_coords = circuit_corners['Y'] - circuit_corners['Y'].min()

# === Leaderboards & Standings ===
st.subheader("🏆 Driver Standings (2025 Season Preview)")
leaderboard_data = {
    "Driver": ["Max Verstappen", "Charles Leclerc", "Lewis Hamilton", "Lando Norris", "Sergio Perez", "George Russell", "Carlos Sainz", "Oscar Piastri"],
    "Team": ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren", "Red Bull Racing", "Mercedes", "Ferrari", "McLaren"],
    "Points": [300, 285, 275, 250, 240, 230, 220, 210]
}
leaderboard_df = pd.DataFrame(leaderboard_data)
st.dataframe(leaderboard_df, use_container_width=True)

# === Circuit Track Animation ===
st.subheader("📍 Silverstone Circuit Animation")

track_fig = go.Figure()

# Plot the track
track_fig.add_trace(go.Scatter(
    x=x_coords,
    y=y_coords,
    mode='lines',
    line=dict(color="white", width=3),
    name="Circuit Path"
))

# Simulated telemetry for 3 drivers
drivers_on_track = ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc"]
driver_colors = ["cyan", "gold", "red"]
laps = 5

# Generate random telemetry
for driver, color in zip(drivers_on_track, driver_colors):
    path_x = np.interp(np.linspace(0, len(x_coords)-1, 100), np.arange(len(x_coords)), x_coords)
    path_y = np.interp(np.linspace(0, len(y_coords)-1, 100), np.arange(len(y_coords)), y_coords)
    path_x += np.random.normal(0, 20, size=path_x.shape)
    path_y += np.random.normal(0, 20, size=path_y.shape)

    track_fig.add_trace(go.Scatter(
        x=path_x,
        y=path_y,
        mode='lines+markers',
        line=dict(color=color, width=2),
        name=f"{driver} Lap Path"
    ))

track_fig.update_layout(
    template=plotly_template,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    height=600,
    showlegend=True
)

st.plotly_chart(track_fig, use_container_width=True)

# === Live Telemetry ===
st.subheader("📡 Live Telemetry (Simulated)")

def simulate_telemetry():
    data = {
        "Driver": [],
        "Speed (km/h)": [],
        "Throttle (%)": [],
        "Brake (%)": [],
        "DRS": []
    }
    for driver in drivers_on_track:
        data["Driver"].append(driver)
        data["Speed (km/h)"].append(random.randint(280, 340))
        data["Throttle (%)"].append(random.randint(60, 100))
        data["Brake (%)"].append(random.randint(0, 30))
        data["DRS"].append(random.choice(["ON", "OFF"]))
    return pd.DataFrame(data)

telemetry_df = simulate_telemetry()
st.table(telemetry_df)

# === Tire Degradation / Fuel Load / Lap Time Graphs ===
st.markdown("---")
col1, col2 = st.columns(2)

# Tire Wear Plot
with col1:
    st.subheader("🛞 Tire Degradation Over Race")
    laps_race = np.arange(1, 51)
    tire_wear = np.maximum(100 - laps_race * random.uniform(1.2, 2.0), 0)

    fig_tire = go.Figure(go.Scatter(
        x=laps_race,
        y=tire_wear,
        mode="lines+markers",
        line=dict(color="orange", width=3)
    ))

    fig_tire.update_layout(
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        xaxis_title="Lap",
        yaxis_title="Tire Wear (%)",
        height=400
    )
    st.plotly_chart(fig_tire, use_container_width=True)

# Fuel Load Plot
with col2:
    st.subheader("⛽ Fuel Load Over Race")
    fuel_load = np.maximum(100 - laps_race * random.uniform(1.5, 2.5), 0)

    fig_fuel = go.Figure(go.Scatter(
        x=laps_race,
        y=fuel_load,
        mode="lines+markers",
        line=dict(color="yellow", width=3)
    ))

    fig_fuel.update_layout(
        template=plotly_template,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        xaxis_title="Lap",
        yaxis_title="Fuel Load (%)",
        height=400
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

# === Pit Stop Strategy (Simulated) ===
st.subheader("🔧 Pit Stop Strategy")
pit_laps = sorted(random.sample(range(5, 50), 3))

fig_pits = go.Figure()
fig_pits.add_trace(go.Scatter(
    x=pit_laps,
    y=[20 for _ in pit_laps],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Pit Stops'
))

fig_pits.update_layout(
    template=plotly_template,
    paper_bgcolor=background_color,
    plot_bgcolor=background_color,
    xaxis_title='Lap',
    yaxis_title='Pit Stop Time (s)',
    height=400
)

st.plotly_chart(fig_pits, use_container_width=True)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.info("Developed for F1 Strategy Presentation | Silverstone Circuit | 2025 Season")
