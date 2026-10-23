
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from raman_simulation import simulation as hop

from threading import Lock, Thread
from queue import Queue
import json
import numpy as np


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Raman Calculator API",
    version="0.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# Initialize simulation ONCE
# --------------------------------------------------

print("Initializing Raman simulation...")

sim = hop.sim(ls_order=8)
sim.get_f()

simulation_lock = Lock()

print("Raman simulation initialized successfully.")


# --------------------------------------------------
# Input parameters
# --------------------------------------------------

class StarkShiftParameters(BaseModel):
    species: str = "40Ca"

    wavelength_nm: float = Field(default=976.0, gt=0)
    beam_detuning_mhz: float = 5.0
    zeeman_mhz: float = 2.63

    red_power_mw: float = Field(default=100.0, ge=0)
    blue_power_mw: float = Field(default=100.0, ge=0)

    red_waist_um: float = Field(default=30.0, gt=0)
    blue_waist_um: float = Field(default=30.0, gt=0)

    red_sigma_plus: float = 0.0
    red_pi: float = 0.0

    blue_sigma_plus: float = 1 / 3
    blue_pi: float = 1 / 3

    lightshift_enabled: bool = False
    ls_power_mw: float = Field(default=100.0, ge=0)
    ls_waist_um: float = Field(default=30.0, gt=0)
    ls_sigma_plus: float = 0.0
    ls_pi: float = 0.0
    ls_wavelength_nm: float = Field(default=976.0, gt=0)


class SimulationParameters(StarkShiftParameters):
    initial_state: int = Field(default=0, ge=0, le=5)
    final_state: int = Field(default=3, ge=0, le=5)


# --------------------------------------------------
# State labels
# --------------------------------------------------

STATES = [
    "+5/2",
    "+3/2",
    "+1/2",
    "-1/2",
    "-3/2",
    "-5/2"
]


# --------------------------------------------------
# Validate input parameters
# --------------------------------------------------

def validate_parameters(p: StarkShiftParameters):

    if p.species not in ("40Ca", "88Sr", "138Ba"):
        raise HTTPException(
            status_code=422,
            detail="Unsupported ion species"
        )

    polarizations = [
        (p.red_sigma_plus, p.red_pi),
        (p.blue_sigma_plus, p.blue_pi),
    ]
    if p.lightshift_enabled:
        polarizations.append((p.ls_sigma_plus, p.ls_pi))

    if any(
        not (
            0 <= sp <= 1
            and 0 <= pi <= 1
            and sp + pi <= 1
        )
        for sp, pi in polarizations
    ):
        raise HTTPException(
            status_code=422,
            detail="Invalid polarization fractions"
        )


# --------------------------------------------------
# Configure shared simulation
#
# Call only while holding simulation_lock.
# --------------------------------------------------

def configure_simulation(p: StarkShiftParameters):

    sim.ls_order = 8

    # Species
    sim.specie = p.species

    # Beam powers (mW -> W)
    sim.Pr = p.red_power_mw * 1e-3
    sim.Pb = p.blue_power_mw * 1e-3
    sim.Pls = p.ls_power_mw * 1e-3 if p.lightshift_enabled else 0.0

    # Beam waists (um -> m)
    sim.w0r = p.red_waist_um * 1e-6
    sim.w0b = p.blue_waist_um * 1e-6
    sim.w0ls = p.ls_waist_um * 1e-6

    # Beam 1 polarization
    sim.r_sp = p.red_sigma_plus
    sim.r_pi = p.red_pi
    sim.r_sm = 1 - sim.r_sp - sim.r_pi

    # Beam 2 polarization
    sim.b_sp = p.blue_sigma_plus
    sim.b_pi = p.blue_pi
    sim.b_sm = 1 - sim.b_sp - sim.b_pi

    # Optional lightshift beam polarization
    sim.ls_sp = p.ls_sigma_plus
    sim.ls_pi = p.ls_pi
    sim.ls_sm = 1 - sim.ls_sp - sim.ls_pi

    # Beam frequency difference (MHz -> rad/s)
    sim.ω = p.beam_detuning_mhz * 2 * np.pi * 1e6

    # Zeeman splitting (MHz -> rad/s)
    sim.ω0 = p.zeeman_mhz * 2 * np.pi * 1e6

    # Species-dependent P-state resonance wavelengths
    resonances = {
        "40Ca": 854.0,
        "88Sr": 1033.0,
        "138Ba": 614.0
    }

    # Laser detuning from P state (THz)
    detuning_thz = 299792.458 * (
        1 / p.wavelength_nm
        - 1 / resonances[p.species]
    )

    # THz -> rad/s
    sim.Δ = detuning_thz * 2 * np.pi * 1e12

    ls_detuning_thz = 299792.458 * (
        1 / p.ls_wavelength_nm - 1 / resonances[p.species]
    )
    sim.Δls = ls_detuning_thz * 2 * np.pi * 1e12


# --------------------------------------------------
# API homepage
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Raman Calculator API is running"
    }


# --------------------------------------------------
# Calculate Stark shifts
# --------------------------------------------------

@app.post("/api/stark-shifts")
def calculate_stark_shifts(p: StarkShiftParameters):

    validate_parameters(p)

    with simulation_lock:

        configure_simulation(p)

        sim.analytics([sim.Pb])

        shifts = {}

        for order in (2, 4, 6, 8):
            shifts[str(order)] = [
                float(
                    getattr(sim, f"δ{i}{order}_bank")[0]
                    / (2 * np.pi * 1e3)
                )
                for i in range(6)
            ]

    return {
        "species": p.species,
        "states": STATES,
        "units": "kHz",
        "shifts": shifts
    }


# --------------------------------------------------
# Calculate all Rabi frequencies
# --------------------------------------------------

@app.post("/api/rabi-frequencies")
def calculate_rabi_frequencies(p: StarkShiftParameters):

    validate_parameters(p)

    with simulation_lock:

        configure_simulation(p)

        original_omega = sim.ω
        original_order = sim.ls_order
        original_transition = sim.transition

        try:

            sim.full_analytics()

            # Results are already in kHz.
            rabi_frequencies = {
                str(transition): float(frequency)
                for transition, frequency
                in sim.rabi_freqs.items()
            }

            predicted_resonances = {
                str(key): float(value / (2 * np.pi * 1e6))
                for key, value in sim.ω_pred_2nd.items()
            }

        finally:

            sim.ω = original_omega
            sim.ls_order = original_order
            sim.transition = original_transition

    transitions = []

    for i in range(6):
        for j in range(i + 1, 6):

            state_i = STATES[i]
            state_j = STATES[j]

            key = f"{state_i}<->{state_j}"
            reverse_key = f"{state_j}<->{state_i}"

            frequency = rabi_frequencies.get(
                key,
                rabi_frequencies.get(reverse_key)
            )
            
            resonance_mhz = predicted_resonances.get(
                key,
                predicted_resonances.get(reverse_key)
            )

            if frequency is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing Rabi frequency for {key}"
                )

            delta_m = abs(i - j)

            if delta_m <= 2:
                photon_order = 2
            elif delta_m == 3:
                photon_order = 4
            else:
                photon_order = 6

            pi_time_us = (
                500.0 / abs(frequency)
                if frequency != 0
                else None
            )

            transitions.append({
                "i": i,
                "j": j,
                "state_i": state_i,
                "state_j": state_j,
                "key": key,
                "photon_order": photon_order,
                "rabi_khz": frequency,
                "pi_time_us": pi_time_us,
                "resonance_mhz": resonance_mhz
            })

    return {
        "species": p.species,
        "states": STATES,
        "units": "kHz",
        "transitions": transitions,
        "rabi_frequencies": rabi_frequencies
    }


# --------------------------------------------------
# Helpers for numerical simulation
# --------------------------------------------------

POPULATION_NAMES = [
    "p52",
    "p32",
    "p12",
    "m12",
    "m32",
    "m52"
]

SPECTROSCOPY_NAMES = [
    "p52_res",
    "p32_res",
    "p12_res",
    "m12_res",
    "m32_res",
    "m52_res"
]


def array_to_list(value):
    """Convert NumPy arrays to JSON-compatible lists."""
    return np.asarray(value, dtype=float).tolist()


def get_population_data(names):
    """Read six state-population arrays from the simulation."""

    return {
        state: array_to_list(getattr(sim, name))
        for state, name in zip(STATES, names)
    }


def get_simulation_result(p, transition):
    """Collect the completed spectroscopy and dynamics data."""

    result = {
        "species": p.species,
        "transition": transition,
        "initial_state": p.initial_state,
        "final_state": p.final_state,
        "states": STATES,

        # Resonant dynamics
        "time_us": array_to_list(
            np.asarray(sim.t, dtype=float) * 1e6
        ),

        "populations": get_population_data(
            POPULATION_NAMES
        ),

        # Spectroscopy
        "detuning_mhz": array_to_list(
            np.asarray(sim.ω_bank, dtype=float)
            / (2 * np.pi * 1e6)
        ),

        "spectroscopy": get_population_data(
            SPECTROSCOPY_NAMES
        ),

        "resonance_mhz": float(
            sim.w_num / (2 * np.pi * 1e6)
        ),
        
        "resonance_error_mhz": float(
            sim.err[1] / (2 * np.pi * 1e6)
        ),
        
        "pi_time_us": float(sim.p[0] * 1e6),
        
        "pi_time_error_us": float(sim.error[0] * 1e6),

        # Rabi fit
        "rabi_fit": {
            "time_us": array_to_list(
                np.asarray(sim.fit[0], dtype=float) * 1e6
            ),
            "population": array_to_list(sim.fit[1])
        }
    }

    # Optional spectroscopy fit.
    # The exact attribute depends on the simulation class.
    if hasattr(sim, "fit_res"):
        fit_res = getattr(sim, "fit_res")

        if (
            isinstance(fit_res, (list, tuple))
            and len(fit_res) >= 2
        ):
            result["spectroscopy_fit"] = {
                "detuning_mhz": array_to_list(
                    np.asarray(fit_res[0], dtype=float)
                    / (2 * np.pi * 1e6)
                ),
                "population": array_to_list(fit_res[1])
            }

    return result


# --------------------------------------------------
# Run numerical simulation
# --------------------------------------------------

def run_numerical_simulation(
    p: SimulationParameters,
    progress_callback=None
):

    transition = (
        f"{STATES[p.initial_state]}<->"
        f"{STATES[p.final_state]}"
    )

    with simulation_lock:

        configure_simulation(p)

        original_omega = sim.ω
        original_order = sim.ls_order
        original_transition = sim.transition

        try:

            sim.transition = transition

            if progress_callback:
                progress_callback(
                    "preparing",
                    0,
                    "Calculating analytical Rabi frequencies..."
                )

            sim.full_analytics()

            sim.transition = transition

            if progress_callback:
                progress_callback(
                    "spectroscopy",
                    0,
                    "Starting spectroscopy scan..."
                )

            # These callbacks are called by the original
            # Raman simulation code during its numerical work.
            def spectroscopy_progress(value):
                if progress_callback:
                    progress_callback(
                        "spectroscopy",
                        value,
                        "Running spectroscopy scan..."
                    )

            def dynamics_progress(value):
                if progress_callback:
                    progress_callback(
                        "dynamics",
                        value,
                        "Simulating resonant dynamics..."
                    )

            sim.full_simulate(
                spectroscopy_progress=spectroscopy_progress,
                dynamics_progress=dynamics_progress
            )

            if progress_callback:
                progress_callback(
                    "spectroscopy",
                    100,
                    "Spectroscopy complete"
                )

                progress_callback(
                    "dynamics",
                    100,
                    "Resonant dynamics complete"
                )

            return get_simulation_result(p, transition)

        finally:

            sim.ω = original_omega
            sim.ls_order = original_order
            sim.transition = original_transition


# --------------------------------------------------
# Regular numerical simulation endpoint
#
# Preserves compatibility with the previous app.js.
# --------------------------------------------------

@app.post("/api/simulate-transition")
def simulate_transition(p: SimulationParameters):

    validate_parameters(p)

    if p.initial_state >= p.final_state:
        raise HTTPException(
            status_code=422,
            detail=(
                "The initial state index must be smaller "
                "than the final state index."
            )
        )

    try:
        return run_numerical_simulation(p)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {exc}"
        ) from exc


# --------------------------------------------------
# Normalize progress callback values
# --------------------------------------------------

def normalize_progress(value):
    """
    Convert progress to a percentage.

    Supports callbacks returning:
      - Fraction between 0 and 1
      - Percentage between 0 and 100

    If the original callback uses a different format,
    this function should be adjusted to match it.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(number):
        return 0.0

    if 0 <= number <= 1:
        number *= 100

    return float(max(0, min(100, number)))


# --------------------------------------------------
# Server-Sent Events formatting
# --------------------------------------------------

def sse_message(event_name, data):
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(data, allow_nan=False)}\n\n"
    )


# --------------------------------------------------
# Live numerical simulation endpoint
#
# POST /api/simulate-transition-stream
#
# Returns progress and final result over SSE.
# --------------------------------------------------

@app.post("/api/simulate-transition-stream")
def simulate_transition_stream(p: SimulationParameters):

    validate_parameters(p)

    if p.initial_state >= p.final_state:
        raise HTTPException(
            status_code=422,
            detail=(
                "The initial state index must be smaller "
                "than the final state index."
            )
        )

    events = Queue()

    def progress_callback(stage, value, message):
        events.put((
            "progress",
            {
                "stage": stage,
                "percent": normalize_progress(value),
                "message": message
            }
        ))

    def worker():

        try:
            result = run_numerical_simulation(
                p,
                progress_callback=progress_callback
            )

            events.put(("result", result))

        except Exception as exc:

            events.put((
                "error",
                {
                    "message": str(exc)
                }
            ))

        finally:
            events.put(("finished", {}))

    def event_generator():

        thread = Thread(
            target=worker,
            daemon=True
        )

        thread.start()

        while True:

            event_name, data = events.get()

            if event_name == "finished":
                break

            yield sse_message(event_name, data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# --------------------------------------------------
# Independent manual simulation endpoints
# Reuse sim.simulate(), sim.get_populations(), and existing fitters.
# Automatic full_simulate() is left unchanged.
# --------------------------------------------------

import useful_funcs
from raman_simulation.simulation import StitchedProgressBar


class ManualSpectroscopyParameters(SimulationParameters):
    interrogation_time_us: float = Field(gt=0)
    scan_start_mhz: float
    scan_stop_mhz: float
    scan_points: int = Field(ge=3, le=2000)


class ManualDynamicsParameters(SimulationParameters):
    dynamics_detuning_mhz: float
    total_time_us: float = Field(gt=0)


def run_manual_simulation(p, stage, progress_callback):
    transition = f"{STATES[p.initial_state]}<->{STATES[p.final_state]}"
    with simulation_lock:
        configure_simulation(p)
        original_omega = sim.ω
        original_order = sim.ls_order
        original_transition = sim.transition
        original_pred = sim.pred_eighth_order
        try:
            sim.transition = transition
            # Use the same analytical prediction used for the automatic fit.
            sim.pred_eighth_order = True
            sim.full_analytics()
            sim.pred_eighth_order = False
            sim.transition = transition
            omega_pred = abs(sim.rabi_freqs[transition]) * 2 * np.pi * 1e3
            if not np.isfinite(omega_pred) or omega_pred <= 0:
                raise ValueError("Analytical Rabi frequency must be nonzero for fitting.")
            s0, sf = p.initial_state, p.final_state

            def progress(value):
                if progress_callback:
                    progress_callback(stage, value, f"Running manual {stage}...")

            if stage == "spectroscopy":
                sim.ω_bank = np.linspace(p.scan_start_mhz, p.scan_stop_mhz, p.scan_points) * 2 * np.pi * 1e6
                t = np.linspace(0, p.interrogation_time_us * 1e-6, 100)
                results = {name: [] for name in POPULATION_NAMES}
                for index, omega in enumerate(sim.ω_bank):
                    sim.ω = omega
                    sim.simulate(t, s0, StitchedProgressBar(progress, index, len(sim.ω_bank)) if progress_callback else None)
                    sim.get_populations()
                    for name in POPULATION_NAMES:
                        results[name].append(float(np.asarray(getattr(sim, name))[-1]))
                target = np.asarray(results[POPULATION_NAMES[sf]])
                height = float(np.max(target))
                center = float(sim.ω_bank[np.argmax(target)]) + 1e-6
                halfwidth = abs(float(sim.ω_bank[np.argmin(abs(target - height / 2))]) - center)
                if halfwidth <= 0 or height <= 0:
                    raise ValueError("Unable to fit spectroscopy: scan has no resolvable peak.")
                p0 = [height, center, 1.5 / halfwidth]
                fit_params, sim.fit_res, sim.err = useful_funcs.fitSinc(sim.ω_bank, target, p0)
                sim.w_num = float(fit_params[1])
                return {
                    "states": STATES,
                    "detuning_mhz": array_to_list(sim.ω_bank / (2 * np.pi * 1e6)),
                    "spectroscopy": {state: results[name] for state, name in zip(STATES, POPULATION_NAMES)},
                    "resonance_mhz": sim.w_num / (2 * np.pi * 1e6),
                    "resonance_error_mhz": float(sim.err[1]) / (2 * np.pi * 1e6),
                    "spectroscopy_fit": {
                        "detuning_mhz": array_to_list(np.asarray(sim.fit_res[0]) / (2 * np.pi * 1e6)),
                        "population": array_to_list(sim.fit_res[1])
                    }
                }

            sim.ω = p.dynamics_detuning_mhz * 2 * np.pi * 1e6
            sim.t = np.linspace(0, p.total_time_us * 1e-6, 1000)
            sim.simulate(sim.t, s0, StitchedProgressBar(progress) if progress_callback else None)
            sim.get_populations()
            target = np.asarray(getattr(sim, POPULATION_NAMES[sf]))
            p0 = [np.pi / omega_pred, 0, 1]
            bounds = [(10e-9, -.1, .6), (100000e-6, .1, 1)]
            fit_params, sim.fit, sim.error = useful_funcs.fitSinFull(sim.t, target, p0, bounds)
            return {
                "states": STATES,
                "time_us": array_to_list(sim.t * 1e6),
                "populations": get_population_data(POPULATION_NAMES),
                "pi_time_us": float(fit_params[0] * 1e6),
                "pi_time_error_us": float(sim.error[0] * 1e6),
                "rabi_fit": {
                    "time_us": array_to_list(np.asarray(sim.fit[0]) * 1e6),
                    "population": array_to_list(sim.fit[1])
                }
            }
        finally:
            sim.ω = original_omega
            sim.ls_order = original_order
            sim.transition = original_transition
            sim.pred_eighth_order = original_pred


def manual_stream(p, stage):
    validate_parameters(p)
    if p.initial_state >= p.final_state:
        raise HTTPException(status_code=422, detail="Initial state must precede final state")
    if stage == "spectroscopy" and p.scan_start_mhz >= p.scan_stop_mhz:
        raise HTTPException(status_code=422, detail="Scan start must be less than scan stop")
    events = Queue()

    def progress_callback(current_stage, value, message):
        events.put(("progress", {
            "stage": current_stage,
            "percent": normalize_progress(value),
            "message": message
        }))

    def worker():
        try:
            events.put(("result", run_manual_simulation(p, stage, progress_callback)))
        except Exception as exc:
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(("finished", {}))

    def generate():
        Thread(target=worker, daemon=True).start()
        while True:
            event_name, data = events.get()
            if event_name == "finished":
                break
            yield sse_message(event_name, data)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"
    })


@app.post("/api/manual-spectroscopy-stream")
def manual_spectroscopy_stream(p: ManualSpectroscopyParameters):
    return manual_stream(p, "spectroscopy")


@app.post("/api/manual-dynamics-stream")
def manual_dynamics_stream(p: ManualDynamicsParameters):
    return manual_stream(p, "dynamics")


