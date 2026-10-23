
const API_URL = "http://127.0.0.1:8000";

// --------------------------------------------------
// DOM elements
// --------------------------------------------------

const form = document.getElementById("parameters");
const status = document.getElementById("status");
const calculateButton = document.getElementById("calculate");
const polarizationWarning = document.getElementById("polarization-warning");

const speciesInput = document.getElementById("species");
const wavelengthInput = document.getElementById("wavelength_nm");
const pDetuningInput = document.getElementById("p_detuning_thz");
const lsWavelengthInput = document.getElementById("ls_wavelength_nm");
const lsDetuningInput = document.getElementById("ls_p_detuning_thz");
const lightshiftControls = document.getElementById("lightshift-controls");
function lightshiftEnabled() {
    return document.querySelector('input[name="lightshift-mode"]:checked')?.value === "on";
}


const starkTab = document.getElementById("tab-stark");
const rabiTab = document.getElementById("tab-rabi");
const starkPanel = document.getElementById("panel-stark");
const rabiPanel = document.getElementById("panel-rabi");

const rabiStatus = document.getElementById("rabi-status");

// Radio buttons replacing the old dropdown.
const rabiDisplayModes = document.querySelectorAll(
    'input[name="rabi-display-mode"]'
);

function getRabiDisplayMode() {
    return document.querySelector(
        'input[name="rabi-display-mode"]:checked'
    )?.value || "frequency";
}

const rabiHexagon = document.getElementById("rabi-hexagon");

const selectedTransitionLabel = document.getElementById(
    "selected-transition-label"
);
const selectedRabiFrequency = document.getElementById(
    "selected-rabi-frequency"
);
const selectedPiTime = document.getElementById(
    "selected-pi-time"
);
const selectedResonance = document.getElementById(
    "selected-resonance"
);

const simulateButton = document.getElementById("simulate-transition");
const simulationResults = document.getElementById("simulation-results");
const simulationProgress = document.getElementById("simulation-progress");

const spectroscopyProgress = document.getElementById(
    "spectroscopy-progress"
);
const spectroscopyProgressText = document.getElementById(
    "spectroscopy-progress-text"
);
const dynamicsProgress = document.getElementById(
    "dynamics-progress"
);
const dynamicsProgressText = document.getElementById(
    "dynamics-progress-text"
);
const simulationProgressMessage = document.getElementById(
    "simulation-progress-message"
);

// --------------------------------------------------
// Application state
// --------------------------------------------------

let isCalculating = false;
let isCalculatingRabi = false;
let isSimulating = false;
let polarizationsValid = true;

let rabiData = null;
let selectedTransition = "0-3";

// --------------------------------------------------
// Constants
// --------------------------------------------------

const parameterNames = [
    "species",
    "wavelength_nm",
    "beam_detuning_mhz",
    "zeeman_mhz",
    "red_power_mw",
    "blue_power_mw",
    "red_waist_um",
    "blue_waist_um",
    "red_sigma_plus",
    "red_pi",
    "blue_sigma_plus",
    "blue_pi",
    "ls_power_mw",
    "ls_waist_um",
    "ls_sigma_plus",
    "ls_pi",
    "ls_wavelength_nm"
];

const resonanceWavelengths = {
    "40Ca": 854.0,
    "88Sr": 1033.0,
    "138Ba": 614.0
};

const SPEED_OF_LIGHT_NM_THZ = 299792.458;

const STATE_COLORS = {
    "+5/2": "#2563EB",
    "+3/2": "#E76F51",
    "+1/2": "#16A085",
    "-1/2": "#9333EA",
    "-3/2": "#D99A00",
    "-5/2": "#DB2777"
};

const PHOTON_COLORS = {
    2: "#64748B",
    4: "#2563EB",
    6: "#D946EF"
};

const RABI_STATES = [
    "+5/2",
    "+3/2",
    "+1/2",
    "-1/2",
    "-3/2",
    "-5/2"
];

const HEXAGON_ORDER = [0, 1, 3, 5, 4, 2];
const SVG_NS = "http://www.w3.org/2000/svg";

// --------------------------------------------------
// Read input parameters
// --------------------------------------------------

function getParameters() {
    const parameters = {};

    for (const name of parameterNames) {
        const input = document.getElementById(name);

        if (!input) {
            throw new Error(`Missing input: ${name}`);
        }

        if (name === "species") {
            parameters[name] = input.value;
            continue;
        }

        if (input.value.trim() === "") {
            throw new Error(`Missing value for ${name}`);
        }

        const value = Number(input.value);

        if (!Number.isFinite(value)) {
            throw new Error(`Invalid value for ${name}`);
        }

        parameters[name] = value;
    }

    parameters.lightshift_enabled = lightshiftEnabled();
    return parameters;
}

// --------------------------------------------------
// Linked wavelength / P-state detuning
// --------------------------------------------------

function updatePDetuning() {
    const wavelength = Number(wavelengthInput.value);
    const resonance = resonanceWavelengths[speciesInput.value];

    if (
        wavelengthInput.value.trim() === "" ||
        !Number.isFinite(wavelength) ||
        wavelength <= 0
    ) {
        pDetuningInput.value = "";
        return;
    }

    const detuning =
        SPEED_OF_LIGHT_NM_THZ / wavelength -
        SPEED_OF_LIGHT_NM_THZ / resonance;

    pDetuningInput.value = detuning.toFixed(2);
    pDetuningInput.setCustomValidity("");
}

function updateWavelength() {
    const detuning = Number(pDetuningInput.value);
    const resonance = resonanceWavelengths[speciesInput.value];

    if (
        pDetuningInput.value.trim() === "" ||
        !Number.isFinite(detuning)
    ) {
        wavelengthInput.value = "";
        pDetuningInput.setCustomValidity(
            "Enter a valid P-state detuning."
        );
        return;
    }

    const frequency =
        SPEED_OF_LIGHT_NM_THZ / resonance + detuning;

    if (frequency <= 0) {
        wavelengthInput.value = "";
        pDetuningInput.setCustomValidity(
            "The resulting laser frequency must be positive."
        );
        pDetuningInput.reportValidity();
        return;
    }

    pDetuningInput.setCustomValidity("");

    const wavelength = SPEED_OF_LIGHT_NM_THZ / frequency;

    wavelengthInput.value = Number(
        wavelength.toPrecision(12)
    );
}

function updateLsDetuning() {
    const wavelength = Number(lsWavelengthInput.value);
    const resonance = resonanceWavelengths[speciesInput.value];
    if (lsWavelengthInput.value.trim() === "" || !Number.isFinite(wavelength) || wavelength <= 0) {
        lsDetuningInput.value = "";
        return;
    }
    const detuning = SPEED_OF_LIGHT_NM_THZ / wavelength - SPEED_OF_LIGHT_NM_THZ / resonance;
    lsDetuningInput.value = detuning.toFixed(2);
    lsDetuningInput.setCustomValidity("");
}

function updateLsWavelength() {
    const detuning = Number(lsDetuningInput.value);
    const resonance = resonanceWavelengths[speciesInput.value];
    if (lsDetuningInput.value.trim() === "" || !Number.isFinite(detuning)) {
        lsWavelengthInput.value = "";
        lsDetuningInput.setCustomValidity("Enter a valid P-state detuning.");
        return;
    }
    const frequency = SPEED_OF_LIGHT_NM_THZ / resonance + detuning;
    if (frequency <= 0) {
        lsWavelengthInput.value = "";
        lsDetuningInput.setCustomValidity("The resulting laser frequency must be positive.");
        lsDetuningInput.reportValidity();
        return;
    }
    lsDetuningInput.setCustomValidity("");
    lsWavelengthInput.value = Number((SPEED_OF_LIGHT_NM_THZ / frequency).toPrecision(12));
}

function updateLightshiftMode() {
    lightshiftControls.hidden = !lightshiftEnabled();
    // Preserve entered values when toggling off; backend forces Pls=0.
    updatePolarizations();
}

// --------------------------------------------------
// Polarization validation
// --------------------------------------------------

function updatePolarizations() {
    const invalidBeams = [];

    for (const beam of ["red", "blue", ...(lightshiftEnabled() ? ["ls"] : [])]) {
        const sigmaPlusInput = document.getElementById(
            `${beam}_sigma_plus`
        );
        const piInput = document.getElementById(
            `${beam}_pi`
        );
        const sigmaMinusInput = document.getElementById(
            `${beam}_sigma_minus`
        );

        const sigmaPlus = Number(sigmaPlusInput.value);
        const pi = Number(piInput.value);

        const validNumbers =
            sigmaPlusInput.value.trim() !== "" &&
            piInput.value.trim() !== "" &&
            Number.isFinite(sigmaPlus) &&
            Number.isFinite(pi);

        const valid =
            validNumbers &&
            sigmaPlus >= 0 &&
            sigmaPlus <= 1 &&
            pi >= 0 &&
            pi <= 1 &&
            sigmaPlus + pi <= 1 + 1e-12;

        sigmaMinusInput.value = validNumbers
            ? (1 - sigmaPlus - pi).toFixed(3)
            : "";

        for (const input of [
            sigmaPlusInput,
            piInput,
            sigmaMinusInput
        ]) {
            input.classList.toggle(
                "invalid-polarization",
                !valid
            );
        }

        if (!valid) {
            invalidBeams.push(
                beam === "red" ? "Beam 1" : beam === "blue" ? "Beam 2" : "Lightshift beam"
            );
        }
    }

    polarizationsValid = invalidBeams.length === 0;

    if (!polarizationsValid) {
        polarizationWarning.textContent =
            `${invalidBeams.join(" and ")}: ` +
            "σ⁺ and π must each be between 0 and 1, " +
            "and σ⁺ + π cannot exceed 1.";

        polarizationWarning.hidden = false;
    } else {
        polarizationWarning.textContent = "";
        polarizationWarning.hidden = true;
    }

    calculateButton.disabled =
        isCalculating || !polarizationsValid;

    simulateButton.disabled =
        isSimulating || !polarizationsValid || getSimulationMode() === "manual";
    for (const id of ["manual-spectroscopy-button", "manual-dynamics-button"]) {
        const button = document.getElementById(id);
        if (button) button.disabled = isSimulating || !polarizationsValid;
    }

    return polarizationsValid;
}

// --------------------------------------------------
// Stark-shift plots
// --------------------------------------------------

function drawPlot(order, states, values) {
    const trace = {
        x: [...states].reverse(),
        y: [...values].reverse(),
        type: "bar",
        marker: {
            color: "#3978b6"
        },
        hovertemplate:
            "m = %{x}<br>Shift = %{y:.5f} kHz<extra></extra>"
    };

    const layout = {
        margin: {
            l: 65,
            r: 15,
            t: 20,
            b: 55
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
            family: "Inter, Segoe UI, sans-serif",
            color: "#52657c",
            size: 11
        },
        xaxis: {
            title: {
                text: "m<sub>J</sub>"
            },
            showgrid: false
        },
        yaxis: {
            title: {
                text: "a.c. Stark shift (kHz)"
            },
            zeroline: true,
            zerolinecolor: "#9baabd",
            gridcolor: "#edf1f6",
            exponentformat: "power"
        },
        showlegend: false,
        autosize: true
    };

    return Plotly.react(
        `plot${order}`,
        [trace],
        layout,
        {
            responsive: true,
            displaylogo: false
        }
    );
}

async function drawCombinedStarkPlots(data) {
    const states = data.states;

    // Sum all four perturbative orders (kHz).
    const totalShifts = states.map((_, i) =>
        [2, 4, 6, 8].reduce(
            (sum, order) => sum + data.shifts[String(order)][i],
            0
        )
    );

    // Plot 1: Total Stark shifts.
    const totalTrace = {
        x: [...states].reverse(),
        y: [...totalShifts].reverse(),
        type: "bar",
        marker: { color: "#3978b6" },
        hovertemplate:
            "m = %{x}<br>Total shift = %{y:.5f} kHz<extra></extra>"
    };

    const layout = {
        margin: { l: 65, r: 15, t: 20, b: 55 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
            family: "Inter, Segoe UI, sans-serif",
            color: "#52657c",
            size: 11
        },
        xaxis: {
            title: { text: "m<sub>J</sub>" },
            showgrid: false
        },
        yaxis: {
            title: { text: "Total a.c. Stark shift (kHz)" },
            zeroline: true,
            zerolinecolor: "#9baabd",
            gridcolor: "#edf1f6",
            exponentformat: "power"
        },
        showlegend: false,
        autosize: true
    };

    await Plotly.react(
        "plot-total",
        [totalTrace],
        layout,
        { responsive: true, displaylogo: false }
    );

    // Plot 2: Energy difference between neighboring states.
    const zeemanKHz =
        Number(document.getElementById("zeeman_mhz").value) * 1000;

    const pairs = states.slice(0, -1).map(
        (state, i) => `${state} ↔ ${states[i + 1]}`
    );

    const splittings = pairs.map(
        (_, i) =>
            (zeemanKHz + totalShifts[i] - totalShifts[i + 1]) / 1000
    );

    const baseline = Math.min(...splittings) - 0.1 * (
        Math.max(...splittings) - Math.min(...splittings) || 0.001
    );
    
    const splittingTrace = {
        x: [...pairs].reverse(),
    
        // Plotly bar heights must be relative to the baseline.
        y: [...splittings].reverse().map(value => value - baseline),
    
        type: "bar",
        marker: { color: "#3978b6" },
        base: baseline,
    
        // Preserve the actual energy splitting for hover labels.
        customdata: [...splittings].reverse(),
    
        hovertemplate:
            "%{x}<br>Splitting = %{customdata:.6f} MHz<extra></extra>"
    };

    await Plotly.react(
        "plot-splittings",
        [splittingTrace],
        {
            ...layout,
            xaxis: {
                title: { text: "Transition" },
                showgrid: false,
                tickangle: -35,
                automargin: true
            },
            yaxis: {
                ...layout.yaxis,
                title: { text: "Energy splitting (MHz)" },
                autorange: true,
                rangemode: "normal",
                zeroline: false,
                tickformat: "~f"
            },
            margin: { l: 75, r: 15, t: 20, b: 100 }
        },
        { responsive: true, displaylogo: false }
    );
}

// --------------------------------------------------
// Calculate Stark shifts
// --------------------------------------------------

async function calculateShifts(event) {
    if (event) {
        event.preventDefault();
    }

    if (isCalculating || !updatePolarizations()) {
        return;
    }

    isCalculating = true;
    updatePolarizations();
    status.textContent = "Calculating...";

    try {
        const response = await fetch(
            `${API_URL}/api/stark-shifts`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(getParameters())
            }
        );

        if (!response.ok) {
            throw new Error(
                `API error ${response.status}: ` +
                await response.text()
            );
        }

        const data = await response.json();

        for (const order of [2, 4, 6, 8]) {
            await drawPlot(
                order,
                data.states,
                data.shifts[String(order)]
            );
        }
        await drawCombinedStarkPlots(data);

        status.textContent = "Calculation complete";

    } catch (error) {
        console.error(error);
        status.textContent = `Error: ${error.message}`;

    } finally {
        isCalculating = false;
        updatePolarizations();
    }
}

// --------------------------------------------------
// Tab switching
// --------------------------------------------------

function switchTab(which) {
    const showRabi = which === "rabi";

    starkPanel.hidden = showRabi;
    rabiPanel.hidden = !showRabi;

    starkTab.classList.toggle("active", !showRabi);
    rabiTab.classList.toggle("active", showRabi);

    starkPanel.classList.toggle("active", !showRabi);
    rabiPanel.classList.toggle("active", showRabi);

    starkTab.setAttribute(
        "aria-selected",
        String(!showRabi)
    );

    rabiTab.setAttribute(
        "aria-selected",
        String(showRabi)
    );

    if (!showRabi) {
        requestAnimationFrame(() => {
            for (const order of [2, 4, 6, 8, "-total", "-splittings"]) {
                const plot = document.getElementById(
                    `plot${order}`
                );

                if (plot && plot.data) {
                    Plotly.Plots.resize(plot);
                }
            }
        });
    }

    // Resize both simulation plots when returning to Rabi tab.
    if (showRabi) {
        requestAnimationFrame(() => {
            for (const id of [
                "spectroscopy-plot",
                "population-plot"
            ]) {
                const plot = document.getElementById(id);

                if (plot && plot.data) {
                    Plotly.Plots.resize(plot);
                }
            }
        });
    }
}

// --------------------------------------------------
// SVG helpers
// --------------------------------------------------

function svgElement(tag, attributes = {}, textContent = null) {
    const element = document.createElementNS(SVG_NS, tag);

    for (const [key, value] of Object.entries(attributes)) {
        element.setAttribute(key, String(value));
    }

    if (textContent !== null) {
        element.textContent = textContent;
    }

    return element;
}

function transitionId(transition) {
    const i = Math.min(transition.i, transition.j);
    const j = Math.max(transition.i, transition.j);

    return `${i}-${j}`;
}

function formatFrequency(value) {
    return Number.isFinite(value)
        ? `${value.toPrecision(4)} kHz`
        : "—";
}

function formatPiTime(value) {
    return value === null || !Number.isFinite(value)
        ? "∞ µs"
        : `${value.toPrecision(4)} µs`;
}

// --------------------------------------------------
// Selected transition
// --------------------------------------------------

function updateSelectedTransition() {
    const [i, j] = selectedTransition
        .split("-")
        .map(Number);

    selectedTransitionLabel.textContent =
        `${RABI_STATES[i]} ↔ ${RABI_STATES[j]}`;

    const transition = rabiData?.transitions.find(
        t => transitionId(t) === selectedTransition
    );

    selectedRabiFrequency.textContent = transition
        ? `${transition.rabi_khz.toPrecision(3)} kHz`
        : "—";

    selectedPiTime.textContent = transition
        ? (transition.pi_time_us === null
            ? "∞ µs"
            : `${transition.pi_time_us.toPrecision(3)} µs`)
        : "—";
        
    selectedResonance.textContent =
        transition && Number.isFinite(transition.resonance_mhz)
            ? `${transition.resonance_mhz.toFixed(4)} MHz`
            : "—";  
}

// --------------------------------------------------
// Line intersections for hexagon labels
// --------------------------------------------------

function crossingPosition(a, b, c, d) {
    const rx = b.x - a.x;
    const ry = b.y - a.y;

    const sx = d.x - c.x;
    const sy = d.y - c.y;

    const cross = (u, v, w, z) => u * z - v * w;

    const denominator = cross(rx, ry, sx, sy);

    if (Math.abs(denominator) < 1e-8) {
        return null;
    }

    const t = cross(
        c.x - a.x,
        c.y - a.y,
        sx,
        sy
    ) / denominator;

    const u = cross(
        c.x - a.x,
        c.y - a.y,
        rx,
        ry
    ) / denominator;

    if (
        t <= 0.001 ||
        t >= 0.999 ||
        u <= 0.001 ||
        u >= 0.999
    ) {
        return null;
    }

    return t;
}

function labelPosition(
    transition,
    transitions,
    coordinates,
    placedLabels = []
) {
    const a = coordinates[transition.i];
    const b = coordinates[transition.j];

    const dx = b.x - a.x;
    const dy = b.y - a.y;

    const lineLength = Math.hypot(dx, dy);

    // Approximate label dimensions in SVG units.
    const labelHeight = 30;

    // Calculate a fixed positioning width for both display modes.
    const frequencyText = Number(
        transition.rabi_khz
    ).toPrecision(3);
    
    const piTimeText = transition.pi_time_us === null
        ? "∞"
        : Number(transition.pi_time_us).toPrecision(3);
    
    const labelWidth = Math.max(
        44,
        frequencyText.length * 11 + 14,
        piTimeText.length * 11 + 14
    );

    let bestPosition = null;
    let bestScore = -Infinity;

    // Search candidate positions along the transition line.
    for (let k = 0; k <= 100; k++) {
        const t = 0.10 + 0.80 * k / 100;

        const x = a.x + t * dx;
        const y = a.y + t * dy;

        let score = 0;

        // Keep labels away from the state circles.
        for (const point of Object.values(coordinates)) {
            const distance = Math.hypot(
                x - point.x,
                y - point.y
            );

            if (distance < 65) {
                score -= 1000;
            }
        }

        // Keep labels away from previously placed labels.
        for (const label of placedLabels) {
            const separationX =
                Math.abs(x - label.x) -
                (labelWidth + label.width) / 2;

            const separationY =
                Math.abs(y - label.y) -
                labelHeight;

            if (separationX < 10 && separationY < 10) {
                score -= 2000;
            }

            score += Math.min(
                Math.hypot(x - label.x, y - label.y),
                150
            ) * 0.4;
        }

        // Avoid crossings with other transition lines.
        for (const other of transitions) {
            if (other === transition) {
                continue;
            }

            const c = coordinates[other.i];
            const d = coordinates[other.j];

            const crossing = crossingPosition(a, b, c, d);

            if (crossing !== null) {
                const distance =
                    Math.abs(t - crossing) * lineLength;

                if (distance < labelWidth / 2 + 12) {
                    score -= 500;
                }
            }
        }

        // Slight preference for the middle of the line.
        score -= Math.abs(t - 0.5) * 12;

        if (score > bestScore) {
            bestScore = score;

            bestPosition = {
                x,
                y,
                width: labelWidth
            };
        }
    }

    return bestPosition;
}

// --------------------------------------------------
// Draw interactive Rabi hexagon
// --------------------------------------------------


function drawRabiHexagon() {
    const svg = svgElement("svg", {
        viewBox: "-50 -80 860 780",
        preserveAspectRatio: "xMidYMid meet",
        role: "img",
        "aria-label":
            "Interactive hexagon of D5/2 Raman transitions"
    });
    
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.display = "block";

    const coordinates = {};

    HEXAGON_ORDER.forEach((stateIndex, vertexIndex) => {
        const angle =
            Math.PI / 2 - vertexIndex * Math.PI / 3;

        coordinates[stateIndex] = {
            x: 380 + 320 * Math.cos(angle),
            y: 310 - 320 * Math.sin(angle)
        };
    });

    const transitions = rabiData?.transitions || [];

    const maxRabi = Math.max(
        0,
        ...transitions.map(t => Math.abs(t.rabi_khz))
    );

    // Keep the original drawing order for transition lines.
    const ordered = [...transitions].sort(
        (a, b) => a.photon_order - b.photon_order
    );

    // Draw the selected transition last so it appears on top.
    ordered.sort(
        (a, b) =>
            Number(
                transitionId(a) === selectedTransition
            ) -
            Number(
                transitionId(b) === selectedTransition
            )
    );

    // --------------------------------------------------
    // Calculate label positions in a FIXED order.
    //
    // This order does not depend on selectedTransition.
    // Therefore, selecting a line cannot move labels.
    // --------------------------------------------------

    const placedLabels = [];
    const labelPositions = new Map();

    const labelOrder = [...transitions].sort(
        (a, b) =>
            transitionId(a).localeCompare(transitionId(b))
    );

    for (const transition of labelOrder) {
        const pos = labelPosition(
            transition,
            transitions,
            coordinates,
            placedLabels
        );

        labelPositions.set(
            transitionId(transition),
            pos
        );

        placedLabels.push(pos);
    }

    // --------------------------------------------------
    // Draw transitions using the precomputed positions.
    // --------------------------------------------------

    for (const transition of ordered) {
        const a = coordinates[transition.i];
        const b = coordinates[transition.j];

        const selected =
            transitionId(transition) === selectedTransition;

        const color =
            PHOTON_COLORS[transition.photon_order];

        const normalized = maxRabi > 0
            ? Math.abs(transition.rabi_khz) / maxRabi
            : 0;

        const width = 1.8 + 5.2 * Math.sqrt(normalized);

        const group = svgElement("g", {
            tabindex: "0",
            role: "button",
            "aria-label":
                `${transition.state_i} to ${transition.state_j}, ` +
                formatFrequency(transition.rabi_khz)
        });

        group.style.cursor = "pointer";
        group.style.outline = "none";

        // Visible transition line.
        group.appendChild(svgElement("line", {
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
            stroke: color,
            "stroke-width": selected ? width + 2 : width,
            "stroke-linecap": "round",
            opacity: selected ? 1 : 0.48
        }));

        // Wider invisible line for easier clicking.
        group.appendChild(svgElement("line", {
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
            stroke: "transparent",
            "stroke-width": 18
        }));

        // Retrieve the fixed position.
        // Do NOT recalculate labelPosition() here.
        const pos = labelPositions.get(
            transitionId(transition)
        );

        // Display Rabi frequency or pi time.
        const value =
            getRabiDisplayMode() === "pi-time"
                ? (
                    transition.pi_time_us === null
                        ? "∞"
                        : Number(
                            transition.pi_time_us
                        ).toPrecision(3)
                )
                : Number(
                    transition.rabi_khz
                ).toPrecision(3);

        const labelWidth = Math.max(
            44,
            value.length * 11 + 14
        );

        // White background behind numerical label.
        group.appendChild(svgElement("rect", {
            x: pos.x - labelWidth / 2,
            y: pos.y - 15,
            width: labelWidth,
            height: 30,
            rx: 11,
            fill: "#FFFFFF",
            stroke: selected ? color : "#E2E8F0",
            "stroke-width": selected ? 1.6 : 0.8
        }));

        // Numerical label.
        group.appendChild(svgElement("text", {
            x: pos.x,
            y: pos.y,
            "text-anchor": "middle",
            "dominant-baseline": "central",
            "font-size": 22,
            "font-family": "Inter, Segoe UI, sans-serif",
            "font-weight": selected ? 700 : 500,
            fill: "#25364B"
        }, value));

        // Selecting a transition only changes highlighting.
        const choose = () => {
            selectedTransition = transitionId(transition);
            updateSelectedTransition();
            drawRabiHexagon();
        };

        group.addEventListener("click", choose);

        group.addEventListener("keydown", event => {
            if (
                event.key === "Enter" ||
                event.key === " "
            ) {
                event.preventDefault();
                choose();
            }
        });

        svg.appendChild(group);
    }

    // --------------------------------------------------
    // Draw state circles above transition lines.
    // --------------------------------------------------

    for (const stateIndex of HEXAGON_ORDER) {
        const point = coordinates[stateIndex];
        const state = RABI_STATES[stateIndex];

        const selected = selectedTransition
            .split("-")
            .map(Number)
            .includes(stateIndex);

        const group = svgElement("g");

        group.appendChild(svgElement("circle", {
            cx: point.x,
            cy: point.y,
            r: selected ? 38 : 35,
            fill: STATE_COLORS[state],
            stroke: selected ? "#0F172A" : "#FFFFFF",
            "stroke-width": selected ? 4 : 3,
            opacity: selected ? 1 : 0.82
        }));

        group.appendChild(svgElement("text", {
            x: point.x,
            y: point.y + 6,
            "text-anchor": "middle",
            "font-size": 18,
            "font-family": "Inter, Segoe UI, sans-serif",
            "font-weight": 700,
            fill: "#FFFFFF"
        }, state));

        svg.appendChild(group);
    }

    // Message before calculating Rabi frequencies.
    if (!rabiData) {
        svg.appendChild(svgElement("text", {
            x: 380,
            y: 310,
            "text-anchor": "middle",
            "font-size": 18,
            fill: "#64748B"
        }, "Calculate Rabi frequencies to show transitions"));
    }

    rabiHexagon.replaceChildren(svg);
}


// --------------------------------------------------
// Calculate Rabi frequencies
// --------------------------------------------------

async function calculateRabiFrequencies() {
    if (isCalculatingRabi || !updatePolarizations()) {
        return;
    }

    isCalculatingRabi = true;
    updatePolarizations();
    rabiStatus.textContent = "Calculating...";

    try {
        const response = await fetch(
            `${API_URL}/api/rabi-frequencies`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(getParameters())
            }
        );

        if (!response.ok) {
            throw new Error(
                `API error ${response.status}: ` +
                await response.text()
            );
        }

        const data = await response.json();

        if (
            !Array.isArray(data.transitions) ||
            data.transitions.length !== 15
        ) {
            throw new Error(
                "Expected 15 transitions from the backend."
            );
        }

        rabiData = data;

        drawRabiHexagon();
        updateSelectedTransition();

        rabiStatus.textContent = "Calculation complete";

    } catch (error) {
        console.error(error);
        rabiStatus.textContent =
            `Error: ${error.message}`;

    } finally {
        isCalculatingRabi = false;
        updatePolarizations();
    }
}

// --------------------------------------------------
// Calculate all analytics
// --------------------------------------------------

async function calculateAnalytics(event) {
    event.preventDefault();

    if (isCalculating || isCalculatingRabi || !updatePolarizations()) {
        return;
    }

    await calculateShifts();
    await calculateRabiFrequencies();
}

// --------------------------------------------------
// Simulation progress bar helpers
// --------------------------------------------------

function setProgress(stage, percent) {
    const value = Math.max(
        0,
        Math.min(100, Number(percent) || 0)
    );

    const rounded = Math.round(value);

    if (stage === "spectroscopy") {
        spectroscopyProgress.value = value;
        spectroscopyProgressText.textContent =
            `${rounded}%`;
    }

    if (stage === "dynamics") {
        dynamicsProgress.value = value;
        dynamicsProgressText.textContent =
            `${rounded}%`;
    }
}

function resetSimulationProgress() {
    setProgress("spectroscopy", 0);
    setProgress("dynamics", 0);

    simulationProgressMessage.textContent =
        "Preparing numerical simulation...";
}

// --------------------------------------------------
// Common Plotly layout for simulation graphs
// --------------------------------------------------

function simulationPlotLayout(xAxisTitle) {
    return {
        autosize: true,

        margin: {
            l: 38,
            r: 4,
            t: 4,
            b: 54,
            pad: 0
        },

        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",

        font: {
            family: "Inter, Segoe UI, sans-serif",
            color: "#52657c",
            size: 11
        },

        xaxis: {
            title: {
                text: xAxisTitle,
                standoff: 0
            },
            automargin: false,
            showgrid: true,
            gridcolor: "#edf1f6",
            zeroline: false
        },

        yaxis: {
            title: {
                text: "Population",
                standoff: 8
            },
            automargin: false,
            range: [-0.05, 1.05],
            gridcolor: "#edf1f6",
            zeroline: true,
            zerolinecolor: "#9baabd"
        },

        hovermode: "closest",
        showlegend: false
    };
}

function formatWithError(value, error) {
    if (!Number.isFinite(value) ||
        !Number.isFinite(error) ||
        error <= 0) {
        return "—";
    }

    // One significant figure in the uncertainty.
    const exponent = Math.floor(Math.log10(error));
    const decimals = Math.max(0, -exponent);

    const roundedError = Number(error.toFixed(decimals));
    const roundedValue = value.toFixed(decimals);

    const errorDigits = Math.round(
        roundedError * Math.pow(10, decimals)
    );

    return `${roundedValue}(${errorDigits})`;
}

const SIMULATION_PLOT_CONFIG = {
    responsive: true,
    displaylogo: false
};

// --------------------------------------------------
// Spectroscopy plot
// --------------------------------------------------

async function drawSpectroscopyPlot(data) {
    if (
        !Array.isArray(data.detuning_mhz) ||
        !data.spectroscopy
    ) {
        throw new Error(
            "Spectroscopy data missing from simulation response."
        );
    }

    const traces = data.states.map(state => {
        const populations = data.spectroscopy[state];

        if (!Array.isArray(populations)) {
            throw new Error(
                `Missing spectroscopy population for ${state}`
            );
        }

        return {
            x: data.detuning_mhz,
            y: populations,
            type: "scatter",
            mode: "lines+markers",
            name: state,
            marker: {
                color: STATE_COLORS[state],
                size: 5
            },
            line: {
                color: STATE_COLORS[state],
                width: 1.5
            },
            hovertemplate:
                "Detuning: %{x:.6f} MHz" +
                "<br>Population: %{y:.4f}" +
                `<extra>${state}</extra>`
        };
    });

    if (
        data.spectroscopy_fit &&
        Array.isArray(data.spectroscopy_fit.detuning_mhz) &&
        Array.isArray(data.spectroscopy_fit.population)
    ) {
        traces.push({
            x: data.spectroscopy_fit.detuning_mhz,
            y: data.spectroscopy_fit.population,
            type: "scatter",
            mode: "lines",
            name: "Resonance fit",
            line: {
                color: "#111827",
                width: 2.5,
                dash: "dash"
            },
            hovertemplate:
                "Detuning: %{x:.6f} MHz" +
                "<br>Fit: %{y:.4f}" +
                "<extra>Resonance fit</extra>"
        });
    }

    const layout = simulationPlotLayout(
        "Beam detuning (MHz)"
    );
    
    layout.title = {
        text: `Resonance: ${formatWithError(
            data.resonance_mhz,
            data.resonance_error_mhz
        )} MHz`,
        font: { size: 15 },
        x: 0.5,
        xanchor: "center"
    };
    
    layout.margin.t = 45;

    if (Number.isFinite(data.resonance_mhz)) {
        layout.shapes = [{
            type: "line",
            x0: data.resonance_mhz,
            x1: data.resonance_mhz,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: {
                color: "#111827",
                width: 1.5,
                dash: "dot"
            }
        }];
    }

    await Plotly.react(
        "spectroscopy-plot",
        traces,
        layout,
        SIMULATION_PLOT_CONFIG
    );
}

// --------------------------------------------------
// Resonant dynamics plot
// --------------------------------------------------

async function drawDynamicsPlot(data) {
    if (
        !Array.isArray(data.time_us) ||
        !data.populations
    ) {
        throw new Error(
            "Dynamics data missing from simulation response."
        );
    }

    const traces = data.states.map(state => {
        const populations = data.populations[state];

        if (!Array.isArray(populations)) {
            throw new Error(
                `Missing dynamics population for ${state}`
            );
        }

        return {
            x: data.time_us,
            y: populations,
            type: "scatter",
            mode: "lines",
            name: state,
            line: {
                color: STATE_COLORS[state],
                width: 2
            },
            hovertemplate:
                "Time: %{x:.3f} µs" +
                "<br>Population: %{y:.4f}" +
                `<extra>${state}</extra>`
        };
    });

    if (
        data.rabi_fit &&
        Array.isArray(data.rabi_fit.time_us) &&
        Array.isArray(data.rabi_fit.population)
    ) {
        traces.push({
            x: data.rabi_fit.time_us,
            y: data.rabi_fit.population,
            type: "scatter",
            mode: "lines",
            name: "Rabi fit",
            line: {
                color: "#111827",
                width: 2.5,
                dash: "dash"
            },
            hovertemplate:
                "Time: %{x:.3f} µs" +
                "<br>Fit: %{y:.4f}" +
                "<extra>Rabi fit</extra>"
        });
    }

    const layout = simulationPlotLayout("Time (µs)");
    
    layout.title = {
        text: `π time: ${formatWithError(
            data.pi_time_us,
            data.pi_time_error_us
        )} µs`,
        font: { size: 15 },
        x: 0.5,
        xanchor: "center"
    };
    
    layout.margin.t = 45;
    
    await Plotly.react(
        "population-plot",
        traces,
        layout,
        SIMULATION_PLOT_CONFIG
    );
}

// --------------------------------------------------
// Draw both simulation graphs
// --------------------------------------------------

async function drawSimulationResults(data) {
    if (!Array.isArray(data.states)) {
        throw new Error(
            "Missing state labels from simulation response."
        );
    }

    simulationResults.hidden = false;

    await Promise.all([
        drawSpectroscopyPlot(data),
        drawDynamicsPlot(data)
    ]);

    requestAnimationFrame(() => {
        for (const id of [
            "spectroscopy-plot",
            "population-plot"
        ]) {
            const plot = document.getElementById(id);
    
            if (plot && plot.data) {
                Plotly.Plots.resize(plot);
            }
        }
    });
}

// --------------------------------------------------
// Read Server-Sent Events from POST fetch
// --------------------------------------------------

async function readSimulationStream(response, onEvent) {
    if (!response.body) {
        throw new Error(
            "The browser did not receive a readable response stream."
        );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";

    function processEventBlock(block) {
        const lines = block.split("\n");

        let eventName = "message";
        const dataLines = [];

        for (const line of lines) {
            if (line.startsWith("event:")) {
                eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).trimStart());
            }
        }

        if (dataLines.length === 0) {
            return;
        }

        const text = dataLines.join("\n");
        const data = JSON.parse(text);

        onEvent(eventName, data);
    }

    function processBuffer() {
        buffer = buffer.replace(/\r\n/g, "\n");

        let boundary = buffer.indexOf("\n\n");

        while (boundary !== -1) {
            const block = buffer.slice(0, boundary);

            buffer = buffer.slice(boundary + 2);

            if (block.trim() !== "") {
                processEventBlock(block);
            }

            boundary = buffer.indexOf("\n\n");
        }
    }

    try {
        while (true) {
            const { value, done } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, {
                stream: true
            });

            processBuffer();
        }

        buffer += decoder.decode();
        processBuffer();

        if (buffer.trim() !== "") {
            processEventBlock(buffer.trim());
        }

    } finally {
        reader.releaseLock();
    }
}

// --------------------------------------------------
// Simulate selected Raman transition
// --------------------------------------------------

async function simulateSelectedTransition() {
    if (getSimulationMode() !== "auto" || isSimulating || !updatePolarizations()) {
        return;
    }

    isSimulating = true;
    updatePolarizations();

    simulateButton.textContent = "Simulating...";
    rabiStatus.textContent = "Running numerical simulation...";

    simulationResults.hidden = false;
    resetSimulationProgress();
    spectroscopyProgress.closest(".simulation-progress-item").hidden = false;
    dynamicsProgress.closest(".simulation-progress-item").hidden = false;

    // Show the progress bars above the visualization grid.
    simulationProgress.hidden = false;

    for (const id of [
        "spectroscopy-plot",
        "population-plot"
    ]) {
        const plot = document.getElementById(id);

        if (plot) {
            Plotly.purge(plot);
            plot.replaceChildren();
        }
    }

    const [initial_state, final_state] =
        selectedTransition.split("-").map(Number);

    let receivedResult = false;
    let receivedError = null;
    let resultData = null;

    try {
        const response = await fetch(
            `${API_URL}/api/simulate-transition-stream`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    ...getParameters(),
                    initial_state,
                    final_state
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                `API error ${response.status}: ` +
                await response.text()
            );
        }

        await readSimulationStream(
            response,
            (eventName, data) => {

                if (eventName === "progress") {
                    if (
                        data.stage === "spectroscopy" ||
                        data.stage === "dynamics"
                    ) {
                        setProgress(
                            data.stage,
                            data.percent
                        );
                    }

                    if (typeof data.message === "string") {
                        simulationProgressMessage.textContent =
                            data.message;

                        rabiStatus.textContent =
                            data.message;
                    }
                }

                else if (eventName === "result") {
                    receivedResult = true;
                    resultData = data;
                }

                else if (eventName === "error") {
                    receivedError = new Error(
                        data.message ||
                        "Unknown numerical simulation error."
                    );
                }
            }
        );

        if (receivedError) {
            throw receivedError;
        }

        if (!receivedResult || !resultData) {
            throw new Error(
                "The simulation stream ended without returning results."
            );
        }

        simulationProgressMessage.textContent =
            "Preparing simulation plots...";

        await drawSimulationResults(resultData);

        setProgress("spectroscopy", 100);
        setProgress("dynamics", 100);

        simulationProgressMessage.textContent =
            "Simulation complete.";

        if (Number.isFinite(resultData.resonance_mhz)) {
            rabiStatus.textContent =
                "Simulation complete — resonance: " +
                `${resultData.resonance_mhz.toFixed(5)} MHz`;
        } else {
            rabiStatus.textContent =
                "Simulation complete";
        }

    } catch (error) {
        console.error(error);

        simulationProgressMessage.textContent =
            `Simulation error: ${error.message}`;

        rabiStatus.textContent =
            `Simulation error: ${error.message}`;

    } finally {
        isSimulating = false;
        simulateButton.textContent = "Simulate transition";
        updatePolarizations();
    }
}

// --------------------------------------------------
// Manual spectroscopy and dynamics (independent streams)
// --------------------------------------------------

function getSimulationMode() {
    return document.querySelector('input[name="simulation-mode"]:checked')?.value || "auto";
}

function updateSimulationMode() {
    document.getElementById("manual-simulation-controls").hidden = getSimulationMode() !== "manual";
    updatePolarizations();
}

function readManualNumber(id, { positive = false, integer = false } = {}) {
    const input = document.getElementById(id);
    const value = Number(input.value);
    if (input.value.trim() === "" || !Number.isFinite(value) ||
        (positive && value <= 0) || (integer && !Number.isInteger(value))) {
        throw new Error(`Invalid value: ${input.closest("label")?.textContent.trim() || id}`);
    }
    return value;
}

async function simulateManual(stage) {
    if (isSimulating || getSimulationMode() !== "manual" || !updatePolarizations()) return;

    let extras;
    try {
        if (stage === "spectroscopy") {
            extras = {
                interrogation_time_us: readManualNumber("manual-interrogation-us", { positive: true }),
                scan_start_mhz: readManualNumber("manual-scan-start-mhz"),
                scan_stop_mhz: readManualNumber("manual-scan-stop-mhz"),
                scan_points: readManualNumber("manual-scan-points", { positive: true, integer: true })
            };
            if (extras.scan_start_mhz >= extras.scan_stop_mhz) throw new Error("Scan start must be less than scan stop.");
            if (extras.scan_points < 3 || extras.scan_points > 2000) throw new Error("Scan points must be between 3 and 2000.");
        } else {
            extras = {
                dynamics_detuning_mhz: readManualNumber("manual-dynamics-detuning-mhz"),
                total_time_us: readManualNumber("manual-dynamics-time-us", { positive: true })
            };
        }
    } catch (error) {
        rabiStatus.textContent = `Input error: ${error.message}`;
        return;
    }

    isSimulating = true;
    updatePolarizations();
    simulationResults.hidden = false;
    simulationProgress.hidden = false;
    const progressItem = stage === "spectroscopy" ? spectroscopyProgress.closest(".simulation-progress-item") : dynamicsProgress.closest(".simulation-progress-item");
    const otherItem = stage === "spectroscopy" ? dynamicsProgress.closest(".simulation-progress-item") : spectroscopyProgress.closest(".simulation-progress-item");
    progressItem.hidden = false;
    otherItem.hidden = true;
    setProgress(stage, 0);
    simulationProgressMessage.textContent = `Starting manual ${stage}...`;
    rabiStatus.textContent = `Running manual ${stage}...`;

    const [initial_state, final_state] = selectedTransition.split("-").map(Number);
    let resultData = null;
    let streamError = null;
    try {
        const response = await fetch(`${API_URL}/api/manual-${stage}-stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...getParameters(), initial_state, final_state, ...extras })
        });
        if (!response.ok) throw new Error(`API error ${response.status}: ${await response.text()}`);
        await readSimulationStream(response, (eventName, data) => {
            if (eventName === "progress" && data.stage === stage) {
                setProgress(stage, data.percent);
                if (data.message) {
                    simulationProgressMessage.textContent = data.message;
                    rabiStatus.textContent = data.message;
                }
            } else if (eventName === "result") resultData = data;
            else if (eventName === "error") streamError = new Error(data.message || "Simulation failed");
        });
        if (streamError) throw streamError;
        if (!resultData) throw new Error("Simulation returned no result.");
        if (stage === "spectroscopy") await drawSpectroscopyPlot(resultData);
        else await drawDynamicsPlot(resultData);
        requestAnimationFrame(() => {
            const plot = document.getElementById(stage === "spectroscopy" ? "spectroscopy-plot" : "population-plot");
            if (plot?.data) Plotly.Plots.resize(plot);
        });
        setProgress(stage, 100);
        simulationProgressMessage.textContent = `Manual ${stage} complete.`;
        rabiStatus.textContent = `Manual ${stage} complete.`;
    } catch (error) {
        console.error(error);
        simulationProgressMessage.textContent = `Simulation error: ${error.message}`;
        rabiStatus.textContent = `Simulation error: ${error.message}`;
    } finally {
        isSimulating = false;
        updatePolarizations();
    }
}

// --------------------------------------------------
// Event listeners
// --------------------------------------------------

// Linked wavelength / detuning.
wavelengthInput.addEventListener(
    "input",
    updatePDetuning
);

pDetuningInput.addEventListener(
    "input",
    updateWavelength
);

speciesInput.addEventListener("change", () => {
    updatePDetuning();
    updateLsDetuning();
});
lsWavelengthInput.addEventListener("input", updateLsDetuning);
lsDetuningInput.addEventListener("input", updateLsWavelength);
document.querySelectorAll('input[name="lightshift-mode"]').forEach(radio => {
    radio.addEventListener("change", updateLightshiftMode);
});

// Polarization inputs.
for (const id of [
    "red_sigma_plus",
    "red_pi",
    "blue_sigma_plus",
    "blue_pi",
    "ls_sigma_plus",
    "ls_pi"
]) {
    document.getElementById(id).addEventListener(
        "input",
        updatePolarizations
    );
}

// Calculate all analytics.
form.addEventListener(
    "submit",
    calculateAnalytics
);

// Results tabs.
starkTab.addEventListener(
    "click",
    () => switchTab("stark")
);

rabiTab.addEventListener(
    "click",
    () => switchTab("rabi")
);

// Run numerical simulation.
simulateButton.addEventListener(
    "click",
    simulateSelectedTransition
);

// Manual simulation mode and buttons.
document.querySelectorAll('input[name="simulation-mode"]').forEach(radio => {
    radio.addEventListener("change", updateSimulationMode);
});
document.getElementById("manual-spectroscopy-button").addEventListener("click", () => simulateManual("spectroscopy"));
document.getElementById("manual-dynamics-button").addEventListener("click", () => simulateManual("dynamics"));

// Radio buttons: update hexagon labels immediately.
rabiDisplayModes.forEach(radio => {
    radio.addEventListener(
        "change",
        drawRabiHexagon
    );
});

// --------------------------------------------------
// Initialize page
// --------------------------------------------------

// Default to Rabi frequency on page load.
const defaultRabiRadio = document.querySelector(
    'input[name="rabi-display-mode"][value="frequency"]'
);

if (defaultRabiRadio) {
    defaultRabiRadio.checked = true;
}

updatePDetuning();
updateLsDetuning();
updateLightshiftMode();
drawRabiHexagon();
updateSelectedTransition();


/* Resize supplementary plots when expanded */

document.getElementById("stark-breakdown")
    .addEventListener("toggle", function () {

        if (!this.open) return;

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {

                for (const id of ["plot2", "plot4", "plot6", "plot8"]) {
                    const plot = document.getElementById(id);

                    if (plot && plot.data) {
                        Plotly.Plots.resize(plot);
                    }
                }

            });
        });
    });
