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

    while y >= -1e6 and t < t_max:
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
    
def find_safe_landing_index(xs, ys, landing_height, max_drop):
    """
    Returns the earliest index where the rider can safely contact the landing
    based on equivalent free-fall height.
    """
    for i in range(len(xs)):
        if ys[i] - landing_height <= max_drop:
            return i
    return None

# ----------------------
# Ramp Geometry
# ----------------------

def takeoff_ramp(v0, angle_deg, duration=0.25):
    angle = np.deg2rad(angle_deg)
    length = v0 * duration
    x = np.array([-length * np.cos(angle), 0])
    y = np.array([-length * np.sin(angle), 0])
    return x, y, length

def tangent_landing_ramp(xs, ys, vxs, vys, landing_height, max_drop):
    """
    Builds a landing ramp that begins at the earliest SAFE contact point
    where the rider can reach the landing elevation without exceeding
    max allowable drop.
    """

    # Find first index where trajectory is at or above landing height
    safe_indices = np.where(ys >= landing_height)[0]

    if len(safe_indices) == 0:
        return np.array([]), np.array([])

    # Choose the LAST such index before descent becomes unsafe
    idx = safe_indices[-1]

    x0 = xs[idx]
    y0 = ys[idx]

    vx = vxs[idx]
    vy = vys[idx]

    if abs(vx) < 1e-6:
        return np.array([]), np.array([])

    slope = vy / vx

    # Build ramp backward toward the apex (this is the key fix)
    ramp_length = max_drop * 4
    ramp_x = np.linspace(x0 - ramp_length, x0, 60)
    ramp_y = y0 + slope * (ramp_x - x0)

    return ramp_x, ramp_y


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

jump_feasible = np.max(ys) >= landing_height


# ----------------------
# Minimum speed required to clear landing
# ----------------------

def required_speed_to_clear(angle_deg, landing_height, g):
    angle = np.deg2rad(angle_deg)
    sin2 = np.sin(2 * angle)
    if sin2 <= 0:
        return None
    return np.sqrt((g * abs(landing_height)) / sin2)

v_required = required_speed_to_clear(angle, landing_height, g)

# ----------------------
# Slider Guidance Logic
# ----------------------

problem_slider = None
guidance_message = None

if not jump_feasible:
    if wind_speed < -5:
        problem_slider = "wind"
        guidance_message = "Strong headwind detected — reduce headwind first"
    elif landing_height > hy * 0.9:
        problem_slider = "landing"
        guidance_message = "Landing is near or above apex — lower landing elevation"
    elif v_required and v0 < v_required:
        problem_slider = "speed"
        guidance_message = "Launch speed too low to clear landing"
    else:
        guidance_message = "Try increasing speed or lowering landing elevation"

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

lx, ly = tangent_landing_ramp(
    xs, ys, vxs, vys,
    landing_height,
    max_drop
)


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
# User Guidance (always visible)
# ----------------------

if not jump_feasible:
    st.warning(
        "⚠️ Jump cannot currently clear the landing.\n\n"
        "Suggested first adjustment:\n"
        f"➡️ **{suggestion if suggestion else 'Reduce headwind or increase speed'}**"
    )
else:
    st.success("✅ Jump is physically feasible with current settings.")


# ----------------------
# Feasibility Warning Display
# ----------------------

if not jump_feasible:
    st.error("❌ Cannot clear landing with current settings")

    if v_required:
        if unit_system == "Metric":
            st.warning(
                f"Minimum required launch speed: **{v_required*3.6:.1f} km/hr**"
            )
        else:
            st.warning(
                f"Minimum required launch speed: **{v_required/1.46667:.1f} mi/hr**"
            )

    if guidance_message:
        st.info(f"Suggested adjustment: {guidance_message}")
clearance = np.max(ys) - landing_height

st.metric(
    "Vertical Clearance Above Landing",
    f"{clearance:.2f} {units}",
    help="Must be positive to clear the landing"
)

# ----------------------
# Plot
# ----------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.set_ylim(
    min(-max_drop * 1.5, np.min(ys) - 1),
    max(np.max(ys) * 1.2, 1)
)

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

























