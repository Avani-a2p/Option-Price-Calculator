import pandas as pd
import streamlit as st
st.set_page_config(page_title = "Guide",layout="wide")


st.write(
    """
    ## 📍 What Does This Dashboard Do?
    - 🧮 Calculates the option price using **four different pricing methods**.
    - 🎯 Allows users to select a **stock** and an **option** from the available widgets.
    - 📊 Provides a comprehensive comparison of the results from different models.

    ## 📍 Methods Used for Pricing:
    1. **Black-Scholes-Merton Pricing Model**  
       - A closed-form solution for European options pricing.
    2. **Monte Carlo Simulation**  
       - A simulation-based approach to estimate the price of options.
    3. **Binomial Pricing Model**  
       - A step-by-step lattice model for valuing options.
    4. **Euler-Maruyama Method**  
       - A numerical method to solve stochastic differential equations.
    """
)
