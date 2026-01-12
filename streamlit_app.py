# Bike Jump Simulator – Interactive Projectile Motion & Safety App
# ---------------------------------------------------------------
# Framework: Streamlit (run with: streamlit run app.py)
# Notes:
# - Metric & Imperial fully synchronized via unit conversion
# - Landing ramp is tangent and BELOW trajectory near terminus
# - Apex & terminus coordinates shown (x, y)
# - Free-body force vectors shown at impact

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Session State Defaults
# ----------------------

if "speed_mps" not in st.session_state:
    st.session_state.speed_mps = 20.0  # m/s (~45 mph)
    st.session_state.angle = 28.0
    st.session_state.mass_kg = 120.0
    st.session_state.area_m2 = 0.7
    st.session_state.Cd = 1.0
    st.session_state.wind_mps = -2.0
    st.session_state.wind_angle = 0.0
    st.session_state.unit_system = "Metric"

# ----------------------
# Physics Core
# ----------------------

def simulate(v0, angle_deg, mass, area, Cd, rho, wind_vx, wind_vy, g, dt=0.01):
    ang = np.deg2rad(angle_deg)
    vx, vy = v0*np.cos(ang), v0*np.sin(ang)
    x, y = 0.0, 0.0
    xs, ys, vxs, vys = [x], [y], [vx], [vy]

    while y >= 0:
        vrx, vry = vx - wind_vx, vy - wind_vy
        vr = np.hypot(vrx, vry)
        Fd = 0.5 * rho * Cd * area * vr**2
        ax = -Fd * (vrx/vr) / mass
        ay = -g - Fd * (vry/vr) / mass
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        xs.append(x); ys.append(y)
        vxs.append(vx); vys.append(vy)
    return np.array(xs), np.array(ys), np.array(vxs), np.array(vys)

# ----------------------
# Ramps
# ----------------------

def takeoff_ramp(v0, angle, duration=0.25):
    ang = np.deg2rad(angle)
    L = v0 * duration
    return np.array([-L*np.cos(ang), 0]), np.array([-L*np.sin(ang), 0]), L


def landing_ramp(xs, ys, vxs, vys, drop):
    x0, y0 = xs[-1], ys[-1]
    slope = vys[-1]/vxs[-1]
    x = np.linspace(x0-4*drop, x0, 120)
    y = slope*(x-x0) - (x-x0)**2/(4*drop)
    return x, y

# ----------------------
# UI
# ----------------------

st.set_page_config(layout="centered")
st.markdown("## 🚵 Bike Jump Simulator")

unit = st.radio("Units", ["Metric", "Imperial"], key="unit_system")

# Unit conversions
MPS_TO_MPH = 2.23694
MPS_TO_KPH = 3.6

if unit == "Imperial":
    speed_mph = st.slider("Launch Speed (mph)", 0.0, 80.0,
                           st.session_state.speed_mps*MPS_TO_MPH)
    st.session_state.speed_mps = speed_mph / MPS_TO_MPH
    mass_lb = st.slider("Bike + Rider Weight (lb)", 150.0, 350.0,
                        st.session_state.mass_kg*2.20462)
    st.session_state.mass_kg = mass_lb / 2.20462
else:
    speed_kph = st.slider("Launch Speed (km/h)", 0.0, 130.0,
                           st.session_state.speed_mps*MPS_TO_KPH)
    st.session_state.speed_mps = speed_kph / MPS_TO_KPH
    st.session_state.mass_kg = st.slider("Bike + Rider Mass (kg)", 70.0, 160.0,
                                         st.session_state.mass_kg)

st.session_state.angle = st.slider("Launch Angle (deg)", 5.0, 60.0, st.session_state.angle)
st.session_state.area_m2 = st.slider("Cross-sectional Area (m²)", 0.1, 2.0, st.session_state.area_m2)
st.session_state.Cd = st.slider("Drag Coefficient", 0.5, 1.3, st.session_state.Cd)

st.session_state.wind_mps = st.slider("Wind Speed (m/s)", -20.0, 20.0, st.session_state.wind_mps)
st.session_state.wind_angle = st.slider("Wind Direction (deg)", -180.0, 180.0, st.session_state.wind_angle)

# ----------------------
# Simulation
# ----------------------

g = 9.81
rho = 1.225
wind_vx = st.session_state.wind_mps*np.cos(np.deg2rad(st.session_state.wind_angle))
wind_vy = st.session_state.wind_mps*np.sin(np.deg2rad(st.session_state.wind_angle))

xs, ys, vxs, vys = simulate(st.session_state.speed_mps,
                             st.session_state.angle,
                             st.session_state.mass_kg,
                             st.session_state.area_m2,
                             st.session_state.Cd,
                             rho, wind_vx, wind_vy, g)

apex_i = np.argmax(ys)
xa, ya = xs[apex_i], ys[apex_i]
xt, yt = xs[-1], ys[-1]

trx, try_, Lr = takeoff_ramp(st.session_state.speed_mps, st.session_state.angle)
lx, ly = landing_ramp(xs, ys, vxs, vys, 1.22)

# ----------------------
# Impact Forces
# ----------------------

v_imp = np.hypot(vxs[-1], vys[-1])
KE = 0.5 * st.session_state.mass_kg * v_imp**2
F_contact = KE / 1.22
N_force = st.session_state.mass_kg * g
R_force = np.hypot(F_contact, N_force)

# ----------------------
# Plot
# ----------------------

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xs, ys, lw=2, solid_capstyle='round', label="Flight Path")
ax.plot(trx, try_, lw=2, label=f"Takeoff Ramp ({Lr:.2f} m)")
ax.plot(lx, ly, lw=2, label="Landing Ramp")
ax.scatter([xa, xt], [ya, yt])

ax.text(xa, ya, f" Apex\n({xa:.2f}, {ya:.2f})")
ax.text(xt, yt, f" Terminus\n({xt:.2f}, {yt:.2f})")

# Force vectors
scale = 0.05
ax.arrow(xt, yt, 0, -N_force*scale, head_width=0.5, label="Normal")
ax.arrow(xt, yt, 0, -F_contact*scale, head_width=0.5, label="Impact")
ax.arrow(xt, yt, 0, -R_force*scale, head_width=0.5, label="Resultant")

ax.set_xlabel("Horizontal Distance (m)")
ax.set_ylabel("Height (m)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("trailism.com/jump-design")
