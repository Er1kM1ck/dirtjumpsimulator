# Bike Jump Simulator
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

def simulate_projectile(
    v0, angle_deg, mass, area, Cd, rho, wind_vx, wind_vy, g,
    landing_height=-1e6,
    dt=0.01,
    t_max=30
):
    angle = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)
    x, y, t = 0.0, 0.0, 0.0

    xs, ys, vxs, vys, ts = [x], [y], [vx], [vy], [t]

    while y >= landing_height and t < t_max:
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
    return xs[i], ys[i], i


def landing_point(xs, ys):
    return xs[-1], ys[-1]

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
    """
    Landing ramp that:
    - Is tangent to the trajectory at landing
    - Never exceeds the flight path (guaranteed)
    - Works for step-ups and step-downs
    """

    x0, y0 = xs[-1], ys[-1]
    slope = vys[-1] / vxs[-1]

    # Build ramp domain near landing
    x = np.linspace(x0 - 6 * max_drop, x0, 200)

    # Tangent line
    y_tangent = y0 + slope * (x - x0)

    # Curved ramp below tangent (energy-limited drop)
    y_ramp = y_tangent - (x - x0)**2 / (4 * max_drop)

    # Interpolate actual trajectory for comparison
    y_flight_interp = np.interp(x, xs, ys)

    # HARD SAFETY CONSTRAINT:
    # Ramp can never exceed the flight path
    y_safe = np.minimum(y_ramp, y_flight_interp)

    return x, y_safe

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
st.title("Bike Jump Simulator")
plot_placeholder = st.empty()

unit_system = st.radio("Unit System", ["Metric", "Imperial"])

if unit_system == "Metric":
    g = 9.81
    rho = 1.225
    units = "m"
    max_drop = 1.22

    col1, col2 = st.columns(2)
    with col1:
        v0_kmh = st.slider("Launch Speed (km/hr)", 5.0, 130.0, 80.0)
    with col2:
        angle = st.slider("Launch Angle (degrees)", 5.0, 60.0, 28.0)

    v0 = v0_kmh / 3.6

    col3, col4 = st.columns(2)
    with col3:
        mass = st.slider("Bike + Rider Mass (kg)", 70.0, 150.0, 120.0)
    with col4:
        area = st.slider("Cross-sectional Area (m²)", 0.1, 2.0, 0.7)

else:
    g = 32.17
    rho = 0.00237
    units = "ft"
    max_drop = 4.0

    col1, col2 = st.columns(2)
    with col1:
        v0_mph = st.slider("Launch Speed (mi/hr)", 5.0, 80.0, 30.0)
    with col2:
        angle = st.slider("Launch Angle (degrees)", 5.0, 60.0, 28.0)

    v0 = v0_mph * 1.46667

    col3, col4 = st.columns(2)
    with col3:
        mass_lb = st.slider("Bike + Rider Weight (lb)", 120.0, 300.0, 200.0)
        mass = mass_lb / g
    with col4:
        area = st.slider("Cross-sectional Area (ft²)", 1.0, 22.0, 7.5)

Cd = st.slider("Drag Coefficient", 0.5, 1.3, 1.0)
if unit_system == "Metric":
    wind_speed_kmh = st.slider("Wind Speed (km/hr)", -50.0, 50.0, 0.0)
    wind_speed = wind_speed_kmh / 3.6
else:
    wind_speed_mph = st.slider("Wind Speed (mi/hr)", -30.0, 30.0, 0.0)
    wind_speed = wind_speed_mph * 1.46667

wind_angle = st.slider("Wind Direction (deg, 0=headwind)", -180.0, 180.0, 0.0)

wind_vx = wind_speed * np.cos(np.deg2rad(wind_angle))
wind_vy = wind_speed * np.sin(np.deg2rad(wind_angle))

# ----------------------
# Temporary simulation to determine apex height
# ----------------------

xs_tmp, ys_tmp, _, _, _ = simulate_projectile(
    v0,
    angle,
    mass,
    area,
    Cd,
    rho,
    wind_vx,
    wind_vy,
    g
)

_, apex_height, _ = apex(xs_tmp, ys_tmp)


landing_height = st.slider(
    "Landing Elevation (relative to takeoff)",
    min_value=-10.0,
    max_value=float(apex_height),
    value=min(0.0, float(apex_height)),
    help="Landing height cannot exceed the apex of the jump"
)
xs, ys, vxs, vys, ts = simulate_projectile(
    v0, angle, mass, area, Cd, rho, wind_vx, wind_vy, g,
    landing_height=landing_height
)

hx, hy, h_idx = apex(xs, ys)
tx, ty = xs[-1], ys[-1]

# ----------------------
# Feasibility Check (A)
# ----------------------

jump_feasible = (
    len(xs) > 5 and
    np.max(ys) >= landing_height
)

# ----------------------
# Offending Slider Detection (B)
# ----------------------

primary_issue = None
suggestion = None

if not jump_feasible:
    if wind_vx < -5:
        primary_issue = "Wind Speed"
        suggestion = "Reduce headwind magnitude"
    elif landing_height > hy * 0.9:
        primary_issue = "Landing Elevation"
        suggestion = "Lower landing elevation"
    elif v0 < 10:
        primary_issue = "Launch Speed"
        suggestion = "Increase launch speed"
    else:
        primary_issue = "Launch Angle"
        suggestion = "Increase launch angle slightly"


trx, try_, ramp_len = takeoff_ramp(v0, angle)
lx, ly = tangent_landing_ramp(xs, ys, vxs, vys, max_drop)

v_imp, KE, F_avg, g_force = impact_metrics(mass, vxs, vys, max_drop, g)

# ----------------------
# Landing Zone Safety Classification
# ----------------------

if g_force <= 5:
    landing_color = "green"
    landing_label = "Safe landing zone (< 5 G’s)"
elif g_force <= 10:
    landing_color = "gold"
    landing_label = "Moderate risk zone (5–10 G’s)"
else:
    landing_color = "red"
    landing_label = "High risk zone (> 10 G’s)"

# ----------------------
# Plot
# ----------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(xs, ys, label="Flight Path")
ax.plot(trx, try_, label=f"Takeoff Ramp ({ramp_len:.2f} {units})")
ax.plot(
    lx,
    ly,
    color=landing_color,
    linewidth=3,
    label=landing_label
)
ax.fill_between(
    lx,
    ly,
    ly - max_drop,
    color=landing_color,
    alpha=0.15
)

ax.scatter(hx, hy)
ax.scatter(tx, ty)
ax.text(hx, hy, f" Apex ({hx:.2f}, {hy:.2f})")
ax.text(tx, ty, f" Terminus ({tx:.2f}, {ty:.2f})")

# Apex guide line
ax.axhline(
    y=hy,
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
    label="Apex Height"
)

ax.set_xlabel(f"Horizontal Distance ({units})")
ax.set_ylabel(f"Vertical Height ({units})")


ax.legend()
ax.grid(True)
ax.set_aspect("auto")

plot_placeholder.pyplot(fig)

# ----------------------
# Safety Readout
# ----------------------

st.subheader("Impact & Rider Safety Metrics")

if unit_system == "Metric":
    KE_display = KE
    KE_units = "J"
    F_display = F_avg
    F_units = "N"
    body_weight_force = mass * g
else:
    KE_display = KE / 1.35582          # J → ft-lb
    KE_units = "ft·lb"
    F_display = F_avg                  # already in lbf
    F_units = "lbf"
    body_weight_force = mass * g       # equals weight in lb

force_multiple = F_display / body_weight_force

if unit_system == "Metric":
    impact_speed_display = v_imp * 3.6        # m/s → km/h
    impact_speed_units = "km/h"
else:
    impact_speed_display = v_imp * 0.681818   # ft/s → mi/h
    impact_speed_units = "mi/h"

if unit_system == "Metric":
    v_imp_display = v_imp * 3.6
    speed_units = "km/hr"
else:
    v_imp_display = v_imp / 1.46667
    speed_units = "mi/hr"

st.write(f"Impact speed: **{v_imp_display:.1f} {speed_units}**")

st.write(f"**Kinetic energy at impact:** {KE_display:.1f} {KE_units}")
st.write(f"**Average stopping force:** {F_display:.1f} {F_units}")
st.write(f"**Equivalent rider load:** {force_multiple:.1f} × body weight")
st.write(f"**Equivalent g-force on rider:** {g_force:.2f} G's")

st.markdown(
    """
**Design guidance (rule-of-thumb):**
- 🟢 **< 5 G's** → generally safer / controllable landing  
- 🟡 **5–10 G's** → increasing injury risk  
- 🔴 **> 10 G's** → high risk of serious injury  

*Landing ramps are typically designed to limit effective fall height
and keep rider g-loads below ~5 g where possible.*
"""
)

if g_force > 10:
    st.error("⚠️ High injury risk: excessive g-forces")
elif g_force > 5:
    st.warning("⚠️ Moderate injury risk")
else:
    st.success("✅ Landing forces within safer design range")


















