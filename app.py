# === Imports ===
import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import fastf1
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# === Streamlit Setup ===
st.set_page_config(page_title="🏎️ F1 Race Strategy & Live Telemetry", layout="wide")
st.title("🏎️ F1 Race Strategy & Live Telemetry Dashboard")
st.markdown("---")

# === TEAM & DRIVER SELECTION ===
teams = {
    "Mercedes": {"drivers": ["Lewis Hamilton", "George Russell"], "color": ["#00D2BE", "#FFFFFF"]},
    "Red Bull Racing": {"drivers": ["Max Verstappen", "Sergio Perez"], "color": ["#1E41FF", "#FFD700"]},
    "Ferrari": {"drivers": ["Charles Leclerc", "Carlos Sainz"], "color": ["#DC0000", "#FFFFFF"]},
    "McLaren": {"drivers": ["Lando Norris", "Oscar Piastri"], "color": ["#FF8700", "#FFFFFF"]}
}

selected_team = st.sidebar.selectbox("Select Your Team", list(teams.keys()))
selected_driver = st.sidebar.selectbox("Select Your Driver", teams[selected_team]["drivers"])
team_colors = teams[selected_team]["color"]

# === Team Logo & Driver Photo ===
team_logo_path = f"assets/logos/{selected_team.lower().replace(' ', '_')}.png"
driver_image_path = f"assets/drivers/{selected_driver.lower().replace(' ', '_')}.png"

st.sidebar.image(team_logo_path, caption=selected_team, use_container_width=True)
st.sidebar.image(driver_image_path, caption=selected_driver, use_container_width=True)

# Make sure cache directory exists
if not os.path.exists('./cache'):
    os.makedirs('./cache')

# === FASTF1 DATASET SETUP ===
fastf1.Cache.enable_cache('./cache')  # Optional: Enable caching to speed up loading

year = 2023
race = 'Bahrain Grand Prix'
session_type = st.sidebar.selectbox("Select Session", ['FP1', 'FP2', 'FP3', 'Q', 'R'])

with st.spinner(f"Loading session data for {race} - {session_type}..."):
    session = fastf1.get_session(year, race, session_type)
    session.load()

drivers = session.drivers
driver_dict = {}
for drv in drivers:
    driver_dict[drv] = session.get_driver(drv)['Abbreviation']

selected_driver_abbr = driver_dict.get(selected_driver.split()[-1][:3].upper(), drivers[0])

st.sidebar.markdown(f"**Session Loaded:** {session.event['EventName']} - {session_type}")

# === LIVE TELEMETRY DATA ===
laps = session.laps.pick_driver(selected_driver_abbr)

if laps.empty:
    st.warning("No lap data available for selected driver/session.")
else:
    fastest_lap = laps.pick_fastest()
    st.markdown(f"🏁 Fastest Lap: **{fastest_lap['LapTime']}**")

    telemetry = fastest_lap.get_car_data().add_distance()

    # === Plot Speed Trace ===
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Speed'],
        mode='lines',
        name='Speed (km/h)',
        line=dict(color=team_colors[0], width=3)
    ))
    fig_speed.update_layout(
        title=f"Speed Trace - {selected_driver}",
        xaxis_title='Distance (m)',
        yaxis_title='Speed (km/h)',
        template='plotly_dark',
        height=400
    )

    # === Plot Throttle & Brake ===
    fig_throttle = go.Figure()
    fig_throttle.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Throttle'],
        mode='lines',
        name='Throttle (%)',
        line=dict(color='lime', width=3)
    ))
    fig_throttle.add_trace(go.Scatter(
        x=telemetry['Distance'],
        y=telemetry['Brake'],
        mode='lines',
        name='Brake (%)',
        line=dict(color='red', width=3)
    ))
    fig_throttle.update_layout(
        title=f"Throttle & Brake - {selected_driver}",
        xaxis_title='Distance (m)',
        yaxis_title='%',
        template='plotly_dark',
        height=400
    )

    # === Layout ===
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_speed, use_container_width=True)
    col2.plotly_chart(fig_throttle, use_container_width=True)

    # === LIVE POSITION TRACKING ===
    driver_position_df = laps[['LapNumber', 'Position']]

    fig_position = px.line(driver_position_df, x='LapNumber', y='Position',
                           title=f"{selected_driver}'s Race Position Over Laps",
                           markers=True,
                           template='plotly_dark')
    fig_position.update_layout(
        yaxis=dict(autorange="reversed"),
        height=400
    )
    st.plotly_chart(fig_position, use_container_width=True)

    # === LIVE LEADERBOARD ===
    leaderboard_df = session.laps.groupby('Driver')['LapTime'].min().reset_index().sort_values(by='LapTime')
    leaderboard_df['Driver'] = leaderboard_df['Driver'].map(lambda x: session.get_driver(x)['Abbreviation'])

    st.subheader("🏁 Leaderboard (Fastest Lap Times)")
    st.dataframe(leaderboard_df, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.caption("Developed using FastF1, Plotly & Streamlit ❤️")
