import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
import os

# Enable FastF1 cache (create this directory in your project root!)
CACHE_DIR = './cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# === Streamlit Config ===
st.set_page_config(page_title="🏎️ F1 Live Telemetry - Silverstone", layout="wide")

# === UI Mode Toggle ===
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", True)

if dark_mode:
    theme = 'plotly_dark'
    bg_color = '#0E1117'
    text_color = 'white'
else:
    theme = 'plotly_white'
    bg_color = 'white'
    text_color = 'black'

# === Title and Branding ===
st.markdown(f"<h1 style='text-align: center; color:{text_color};'>🏎️ F1 Live Telemetry Dashboard - Silverstone GP</h1>", unsafe_allow_html=True)
st.image("assets/f1_logo.png", width=150)

# === Load Live Session Data ===
st.sidebar.header("Race Selection")
year = st.sidebar.selectbox("Select Year", [2023], index=0)
gp = st.sidebar.selectbox("Select Grand Prix", ['Silverstone'], index=0)
session_type = st.sidebar.selectbox("Session Type", ['R'], index=0)

st.sidebar.markdown("⚠️ This loads live telemetry data and may take a minute!")

session = fastf1.get_session(year, gp, session_type)
session.load(live_timing_data=True)

# === Driver Selection ===
drivers = session.drivers
driver_names = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

selected_driver = st.sidebar.selectbox("Select Driver for Telemetry", driver_names)

# === Get Driver Telemetry ===
laps = session.laps
laps_driver = laps.pick_driver(selected_driver)
fastest_lap = laps_driver.pick_fastest()

telemetry = fastest_lap.get_car_data().add_distance()

# === Leaderboard DataFrame ===
leaderboard_df = laps[['Driver', 'LapNumber', 'Position', 'LapTime']].sort_values(by='Position')
st.subheader("🏁 Live Leaderboard")
st.dataframe(leaderboard_df)

# === Circuit Map (Animated Driver Positions) ===
st.subheader("🏎️ Circuit Map - Silverstone (Fastest Lap)")

fig_circuit = go.Figure()

# Plot driver path
fig_circuit.add_trace(go.Scatter(
    x=telemetry['X'],
    y=telemetry['Y'],
    mode='lines',
    line=dict(width=3, color='cyan'),
    name=selected_driver
))

# Animated marker for the driver
fig_circuit.add_trace(go.Scatter(
    x=[telemetry['X'].iloc[0]],
    y=[telemetry['Y'].iloc[0]],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Car'
))

fig_circuit.update_layout(
    template=theme,
    title=f"{selected_driver} - Circuit Position (Silverstone)",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=True,
    height=600,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=text_color)
)

st.plotly_chart(fig_circuit, use_container_width=True)

# === Speed, Throttle, Brake Telemetry ===
st.subheader("📈 Telemetry Data")

col1, col2 = st.columns(2)

with col1:
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Speed'],
        mode='lines',
        name='Speed (km/h)',
        line=dict(color='lime')
    ))
    fig_speed.update_layout(
        template=theme,
        title="Speed over Distance",
        xaxis_title='Distance (m)',
        yaxis_title='Speed (km/h)',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with col2:
    fig_throttle = go.Figure()
    fig_throttle.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Throttle'],
        mode='lines',
        name='Throttle (%)',
        line=dict(color='yellow')
    ))
    fig_throttle.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Brake'],
        mode='lines',
        name='Brake (%)',
        line=dict(color='red')
    ))
    fig_throttle.update_layout(
        template=theme,
        title="Throttle & Brake over Distance",
        xaxis_title='Distance (m)',
        yaxis_title='Percentage (%)',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    st.plotly_chart(fig_throttle, use_container_width=True)

# === Lap Times Summary ===
st.subheader(f"🏎️ {selected_driver} Lap Time: {fastest_lap['LapTime']}")

st.success("✅ Live Telemetry Loaded Successfully!")

