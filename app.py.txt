# Interactive Projectile Motion Web App – Mountain Bike Jump Safety Simulator
# ----------------------------------------------------------------------
# Framework: Streamlit (run with: streamlit run app.py)
# Features:
# - Interactive sliders & inputs (metric & imperial)
# - Wind resistance with direction
# - Tangent-matched landing ramp
# - Takeoff ramp length displayed
# - Energy & impact-force calculations for rider safety

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Physics Core
# ----------------------

def simulate_projectile(v0, angle_deg, mass, area, Cd, rho, wind_vx, wind_vy, g, dt=0.01, t_max=30):
    angle = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)
    x, y, t = 0.0, 0.0, 0.0

    xs, ys, vxs, vys, ts = [x], [y], [vx], [vy], [t]

    while y >= 0 and t < t_max:
        vrel_x = vx - wind_vx
        vrel_y = vy - wind_vy
        vrel = np.hypot(vrel_x, vrel_y)

        Fd = 0.5 * rho * Cd * area * vrel**2
        ax = -Fd * (vrel_x / vrel) / mass
        ay = -g - (Fd * (vrel_y / vrel) / mass)

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        ts.append(t)

    return np.array(xs), np.array(ys), np.array(vxs), np.array(vys), np.array(ts)


def apex(xs, ys):
    i = np.argmax(ys)
    return xs[i], ys[i]

# ----------------------
# Ramp Geometry
# ----------------------

def takeoff_ramp(v0, angle_deg, duration=0.25):
    angle = np.deg2rad(angle_deg)
    length = v0 * duration
    x = np.array([-length * np.cos(angle), 0])
    y = np.array([-length * np.sin(angle), 0])
    return x, y, length


def tangent_landing_ramp(xs, ys, vxs, vys, max_drop):
    # Use final trajectory slope
    x0, y0 = xs[-1], ys[-1]
    slope = vys[-1] / vxs[-1]

    x = np.linspace(x0 - max_drop * 4, x0, 120)
    y = slope * (x - x0) + 0.25 * (x - x0)**2 / max_drop
    return x, y

# ----------------------
# Impact & Energy Safety
# ----------------------

def impact_metrics(mass, vxs, vys, max_drop, g):
    v_impact = np.hypot(vxs[-1], vys[-1])
    KE = 0.5 * mass * v_impact**2
    stopping_distance = max_drop
    avg_force = KE / stopping_distance
    g_force = avg_force / (mass * g)
    return v_impact, KE, avg_force, g_force

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(layout="wide")
st.title("Mountain Bike Jump – Projectile Motion & Safety App")

unit_system = st.radio("Unit System", ["Metric", "Imperial"])

if unit_system == "Metric":
    g = 9.81
    rho = 1.225
    v0 = st.slider("Launch Speed (m/s)", 5.0, 40.0, 22.0)
    angle = st.slider("Launch Angle (degrees)", 5.0, 60.0, 28.0)
    mass = st.slider("Bike + Rider Mass (kg)", 70.0, 150.0, 120.0)
    area = st.slider("Cross-sectional Area (m²)", 0.3, 1.2, 0.7)
    max_drop = 1.22
    units = "m"
else:
    g = 32.17
    rho = 0.00237
    v0 = st.slider("Launch Speed (ft/s)", 20.0, 130.0, 72.0)
    angle = st.slider("Launch Angle (degrees)", 5.0, 60.0, 28.0)
    mass = st.slider("Bike + Rider Mass (slug)", 4.0, 8.0, 5.0)
    area = st.slider("Cross-sectional Area (ft²)", 4.0, 12.0, 7.5)
    max_drop = 4.0
    units = "ft"

Cd = st.slider("Drag Coefficient", 0.5, 1.3, 1.0)
wind_speed = st.slider(f"Wind Speed ({units}/s)", -30.0, 30.0, -5.0)
wind_angle = st.slider("Wind Direction (deg, 0=headwind)", -180.0, 180.0, 0.0)

wind_vx = wind_speed * np.cos(np.deg2rad(wind_angle))
wind_vy = wind_speed * np.sin(np.deg2rad(wind_angle))

xs, ys, vxs, vys, ts = simulate_projectile(v0, angle, mass, area, Cd, rho, wind_vx, wind_vy, g)
hx, hy = apex(xs, ys)

trx, try_, ramp_len = takeoff_ramp(v0, angle)
lx, ly = tangent_landing_ramp(xs, ys, vxs, vys, max_drop)

v_imp, KE, F_avg, g_force = impact_metrics(mass, vxs, vys, max_drop, g)

# ----------------------
# Plot
# ----------------------

fig, ax = plt.subplots()
ax.plot(xs, ys, label="Flight Path")
ax.plot(trx, try_, label=f"Takeoff Ramp ({ramp_len:.2f} {units})")
ax.plot(lx, ly, label="Safe Landing Ramp")
ax.scatter(hx, hy)
ax.text(hx, hy, f" Apex ({hx:.2f}, {hy:.2f})")
ax.set_xlabel(f"Horizontal Distance ({units})")
ax.set_ylabel(f"Vertical Height ({units})")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ----------------------
# Safety Readout
# ----------------------

st.subheader("Impact & Safety Metrics")
st.write(f"Impact speed: **{v_imp:.2f} {units}/s**")
st.write(f"Kinetic energy at impact: **{KE:.1f}**")
st.write(f"Average stopping force: **{F_avg:.1f}**")
st.write(f"Equivalent g-force on rider: **{g_force:.2f} g**")

if g_force > 10:
    st.error("⚠️ High injury risk: excessive g-forces")
elif g_force > 5:
    st.warning("⚠️ Moderate injury risk")
else:
    st.success("✅ Landing forces within safer range")
