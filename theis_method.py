import numpy as np
from scipy.optimize import curve_fit
from scipy.special import exp1
import matplotlib.pyplot as plt

def theis_equation(r, t, Q, T, S):
    """
    Theis equation function

    Parameters:
    -- r: float or numpy array, radial distance from well (m)
    -- t: float or numpy array, time since pumping started (s)
    -- Q: float, pumping rate (m^3/s)
    -- T: float, transmissivity (m^2/s)
    -- S: float, storativity (dimensionless)

    Returns:
    -- s: float or numpy array, drawdown (m)
    """
    u = (r**2 * S) / (4 * T * t)
    W_u = np.exp(-u) * exp1(u)
    s = (Q / (4 * np.pi * T)) * W_u
    return s

if __name__ == "__main__":

    # Generate synthetic drawdown data
    np.random.seed(0)  # For reproducibility
    n_points = 100  # Number of data points
    r = 100  # Radial distance from well (m)
    t = np.linspace(1, 3600, n_points)  # Time since pumping started (s)
    Q = 100  # Pumping rate (m^3/s)
    T_true = 1000  # True transmissivity (m^2/s)
    S_true = 0.001  # True storativity (dimensionless)
    s_true = theis_equation(r, t, Q, T_true, S_true)  # True drawdown data
    s_noise = np.random.normal(loc=0, scale=0.001, size=n_points)  # Add noise to drawdown data
    s_data = s_true + s_noise  # Observed drawdown data
  
    # Define the function for curve fitting
    def theis_fit(x, T, S):
        r, t, Q = x
        return theis_equation(r, t, Q, T, S)
    
    # Initial parameter guesses for T and S
    T_guess = 500
    S_guess = 0.0005
    params_guess = [T_guess, S_guess]
    
    # Perform nonlinear regression to estimate T and S
    params_opt, params_cov = curve_fit(
        theis_fit, 
        (np.repeat(r, n_points), t, np.repeat(Q, n_points)), 
        s_data
    )
    
    # Extract the estimated parameters
    T_est = params_opt[0]
    S_est = params_opt[1]
    
    # Print the estimated parameters
    print("Estimated transmissivity (T):", T_est, "m^2/s")
    print("Estimated storativity (S):", S_est, "dimensionless")

    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(t, s_data)
    ax.plot(t, theis_equation(r, t, Q, T_est, S_est))
    
    # add plot labels
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (m)")
    ax.set_title("Drawdown vs. Time for Aquifer Test using Theis Method")
    results = f"Estimated transmissivity (T): {T_est:.2f} $m^2/s$\nEstimated Storativity (S): {S_est:.8f} (dimensionless)"
    ax.text(0.5, 0.1, results, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.show()
