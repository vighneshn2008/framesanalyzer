"""
calculations.py
----------------
Physics calculations for the rolling-body experiment.

Ideal theory:
    a = g*sin(theta) / (1 + I/(m*R^2))          ... (1)

Kinematics (object released from rest, travels distance d in time t):
    d = 0.5 * a * t^2   =>   a = 2d / t^2         ... (2)

Combining (1) and (2), solve for experimental I:
    I/(m*R^2) = g*sin(theta)/a - 1
    I_exp     = m * R^2 * (g*sin(theta)/a - 1)

With losses (rolling friction mu and/or air drag) the driving force along
the incline is reduced:
    mg*sin(theta)  =  (m + I/R^2)*a + F_loss
with
    F_loss = mu*m*g*cos(theta)  +  0.5*rho*C_d*A*v^2

Solving for a loss-corrected I (using the measured a_exp and a mean drag
force evaluated at the average speed v = d/t):
    I_corr = m*R^2 * ( (g*sin(theta) - mu*g*cos(theta) - F_drag/m) / a_exp - 1 )
"""

import math

G = 9.78137        # m/s^2, acceleration due to gravity
AIR_DENSITY = 1.225  # kg/m^3, density of air at sea level


def experimental_acceleration(distance_m, time_s):
    """a = 2d / t^2 (object starts from rest)."""
    if time_s is None or time_s <= 0:
        raise ValueError("time_s must be a positive number")
    if distance_m is None or distance_m <= 0:
        raise ValueError("distance_m must be a positive number")
    return 2 * distance_m / (time_s ** 2)


def experimental_moment_of_inertia(mass_kg, radius_m, theta_deg, a_exp):
    """I_exp = m*R^2 * (g*sin(theta)/a_exp - 1)  (ideal, losses folded in)"""
    if a_exp is None or a_exp <= 0:
        raise ValueError("a_exp must be a positive number")
    theta_rad = math.radians(theta_deg)
    ratio = (G * math.sin(theta_rad) / a_exp) - 1
    return mass_kg * (radius_m ** 2) * ratio


def circle_area(radius_m):
    """Frontal / cross-sectional area of the body, A = pi*R^2."""
    if not radius_m:
        return 0.0
    return math.pi * (radius_m ** 2)


def friction_force(mass_kg, theta_deg, mu):
    """F_friction = mu * m * g * cos(theta)  (rolling resistance / slide loss)."""
    if not mu:
        return 0.0
    return mu * mass_kg * G * math.cos(math.radians(theta_deg))


def drag_force(velocity_mps, cd=None, area_m2=None, rho=AIR_DENSITY):
    """F_drag = 0.5 * rho * C_d * A * v^2  (quadratic air drag)."""
    if not cd or not area_m2:
        return 0.0
    return 0.5 * rho * cd * area_m2 * (velocity_mps ** 2)


def corrected_moment_of_inertia(mass_kg, radius_m, theta_deg, a_exp,
                                mu=0.0, drag_force_n=0.0):
    """
    I_corr = m*R^2 * ( (g*sin(theta) - mu*g*cos(theta) - F_drag/m) / a_exp - 1 )

    Same kinematics as experimental_moment_of_inertia but with the loss
    terms subtracted from the driving force, so the experimental I is NOT
    inflated by friction + air drag. Raises ValueError if the losses exceed
    the driving force (no physical solution).
    """
    if a_exp is None or a_exp <= 0:
        raise ValueError("a_exp must be a positive number")
    if mass_kg <= 0 or radius_m <= 0 or theta_deg is None:
        raise ValueError("mass, radius and theta must all be positive")
    theta_rad = math.radians(theta_deg)
    drive = (G * math.sin(theta_rad)
             - (mu or 0.0) * G * math.cos(theta_rad)
             - (drag_force_n or 0.0) / mass_kg)
    if drive <= 0:
        raise ValueError("losses exceed the driving force")
    ratio = (drive / a_exp) - 1
    return mass_kg * (radius_m ** 2) * ratio


def percent_error(experimental, theoretical):
    if experimental is None or theoretical in (None, 0):
        return None
    return abs(experimental - theoretical) / abs(theoretical) * 100.0


def average_time_ms(trials_ms):
    """Average of whichever of the (up to 3) trial times are filled in."""
    valid = [t for t in trials_ms if t is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)
