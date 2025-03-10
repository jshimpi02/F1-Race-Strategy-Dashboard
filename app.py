import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Simple demo app (replace with your race app)
st.title('🏎️ F1 Race Strategy Simulator')
st.write('This is a test dashboard!')

# Plot example
fig, ax = plt.subplots()
ax.plot(np.random.randn(50).cumsum())
st.pyplot(fig)
