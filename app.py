import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from raman_simulation import simulation as hop
import plotly.graph_objects as go
import time

class App:
    def __init__(self):
        st.set_page_config(page_title="HOP Calculator", layout="wide")
        st.markdown(
            "<h1 style='margin-bottom:10px;'>D<sub>5/2</sub> High-Order Calculator</h1>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <style>
        div[data-testid="stPills"] button {
            width: 110px !important;
            height: 90px !important;
            font-size: 24px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        species = st.pills(
            "Ion species",
            ["⁴⁰Ca⁺", "⁸⁸Sr⁺", "¹³⁸Ba⁺"],
            default="⁴⁰Ca⁺",
            selection_mode="single",
            label_visibility="collapsed",
            key="species_selector"
        )

        species_dict = {
            "⁴⁰Ca⁺": "40Ca",
            "⁸⁸Sr⁺": "88Sr",
            "¹³⁸Ba⁺": "138Ba"
        }

        with st.spinner("Initializing calculator..."):
            self.sim = self.initialize()
            self.sim.specie = species_dict[species]
            
        self.inputs()

        # automatically recalculates Stark shifts whenever an input changes
        self.sim.analytics([self.sim.Pb])

        tab1, tab2 = st.tabs(["Stark Shifts", "Rabi Frequencies"])

        with tab1:
            self.plot_zeeman_levels()
            self.plot_stark_shifts()
        with tab2:
            if "rabi" not in st.session_state:
                self.calculate_rabi()

            calculate, simulate = self.plot_rabi_frequencies()

            if calculate:
                self.calculate_rabi()
                st.rerun()
                    
            if simulate:
                self.sim.full_analytics()

                st.write("Spectroscopy")
                spec_bar = st.progress(0)

                st.write("Resonant dynamics")
                dyn_bar = st.progress(0)

                timers = {"spec": None, "dyn": None}

                def update_bar(bar, x, key):
                    if timers[key] is None:
                        timers[key] = time.time()

                    elapsed = time.time() - timers[key]
                    remaining = elapsed*(1-x)/x if x > 0 else 0
                    mins, secs = divmod(int(remaining), 60)

                    text = "Complete" if x >= 1 else f"Estimated time remaining: {mins}m {secs}s"
                    bar.progress(min(int(100*x), 100), text=text)

                self.sim.full_simulate(
                    spectroscopy_progress=lambda x: update_bar(spec_bar, x, "spec"),
                    dynamics_progress=lambda x: update_bar(dyn_bar, x, "dyn")
                )
                
                spec_bar.empty()
                dyn_bar.empty()

                self.save_simulation()
                st.session_state.next_beam_detuning = float(self.sim.w_num/(2*np.pi*1e6))
                self.sim.ω = self.sim.w_num
                st.rerun()

            if "simulation" in st.session_state:
                self.plot_simulation()
                    

    def initialize(_self):
        sim = hop.sim(ls_order=8)
        sim.__dict__.update(_self.get_f_cached())
        return sim

    @st.cache_resource
    def get_f_cached(_self):
        sim = hop.sim(ls_order=8)
        sim.get_f()
        return sim.__dict__

    def inputs(self):
        st.sidebar.header("Parameters")

        # Column titles
        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        col1.markdown("**Beam 1**")
        col2.markdown("**Beam 2**")

        # Power
        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        label.write("Power")
        # self.sim.Pr = col1.number_input("Pr", value=100.0, label_visibility="collapsed")*1e-3
        # self.sim.Pb = col2.number_input("Pb", value=100.0, label_visibility="collapsed")*1e-3
        self.sim.Pr = col1.slider("Pr", min_value=0.0, max_value=300.0, value=100.0, step=1.0, label_visibility="collapsed")*1e-3
        self.sim.Pb = col2.slider("Pb", min_value=0.0, max_value=300.0, value=100.0, step=1.0, label_visibility="collapsed")*1e-3
        
        # Beam waist
        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        label.write("Waist (μm)")
        self.sim.w0r = col1.number_input("w0r", format="%.1f", value=30.0, step=1.0, label_visibility="collapsed")*1e-6
        self.sim.w0b = col2.number_input("w0b", format="%.1f", value=30.0, step=1.0, label_visibility="collapsed")*1e-6
        
        # σ+
        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        label.write("σ+")
        self.sim.r_sp = col1.number_input("spr", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed")
        self.sim.b_sp = col2.number_input("spb", value=1/3, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed")

        # π
        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        label.write("π")
        self.sim.r_pi = col1.number_input("pir", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed")
        self.sim.b_pi = col2.number_input("pib", value=1/3, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed")

        # σ− (calculated)
        self.sim.r_sm = 1 - self.sim.r_sp - self.sim.r_pi
        self.sim.b_sm = 1 - self.sim.b_sp - self.sim.b_pi

        label, col1, col2 = st.sidebar.columns([1, 2, 2])
        label.write("σ−")
        col1.write(f"{self.sim.r_sm:.2f}")
        col2.write(f"{self.sim.b_sm:.2f}")

        # Additional parameters
        st.sidebar.divider()
        if "next_beam_detuning" in st.session_state:
            st.session_state.beam_detuning = st.session_state.pop("next_beam_detuning")

        if "beam_detuning" not in st.session_state:
            st.session_state.beam_detuning = 5.0
        self.sim.ω = st.sidebar.slider("Beam detuning (MHz)", min_value=-20.0, max_value=20.0, step=0.1, key="beam_detuning")*2*np.pi*1e6        

        
        # P3/2 resonance wavelengths (nm)
        resonance_nm = {"40Ca": 854.0, "88Sr": 1033.0, "138Ba": 614.0}
        λ0 = resonance_nm[self.sim.specie]
        c_nm_THz = 299792.458
        f0 = c_nm_THz/λ0

        # Initialize laser wavelength once
        if "laser_wavelength" not in st.session_state:
            st.session_state.laser_wavelength = 976.0

        # Recalculate detuning when species changes, keeping wavelength fixed
        if st.session_state.get("detuning_species") != self.sim.specie:
            st.session_state.detuning_species = self.sim.specie
            st.session_state.p_detuning = c_nm_THz/st.session_state.laser_wavelength - f0

        def detuning_changed():
            Δf = st.session_state.p_detuning
            st.session_state.laser_wavelength = c_nm_THz/(f0 + Δf)

        def wavelength_changed():
            λ = st.session_state.laser_wavelength
            st.session_state.p_detuning = c_nm_THz/λ - f0

        col1, col2 = st.sidebar.columns(2)

        Δ_THz = col1.number_input(
            "P state detuning (THz)",
            step=1.0,
            format="%.1f",
            key="p_detuning",
            on_change=detuning_changed
        )

        col2.number_input(
            "Laser wavelength (nm)",
            step=1.0,
            format="%.0f",
            key="laser_wavelength",
            on_change=wavelength_changed
        )

        self.sim.Δ = Δ_THz*2*np.pi*1e12

        
        
        self.sim.ω0 = st.sidebar.number_input("Zeeman splitting (MHz)", value=2.63)*2*np.pi*1e6
        self.sim.get_rabi_frequencies()

    def calculate_rabi(self):
        self.sim.full_analytics()
        st.session_state.rabi = {
            key: value
            for key,value in self.sim.rabi_freqs.items()
        }

    def plot_stark_shifts(self):
        labels = ["-5/2", "-3/2", "-1/2", "+1/2", "+3/2", "+5/2"]
        state_colors = {
            "+5/2": "#636EFA", "+3/2": "#EF553B", "+1/2": "#00CC96",
            "-1/2": "#AB63FA", "-3/2": "#FFA15A", "-5/2": "#19D3F3"
        }
        colors = {
            2: "gray",
            4: "gray",
            6: "gray",
            8: "gray"
        }

        titles = {
            2: "black",
            4: "black",
            6: "black",
            8: "black"
        }

        def make_fig(values, title, color):
            fig = go.Figure()
            fig.add_bar(x=labels, y=values[::-1], marker_color=[state_colors[x] for x in labels], hovertemplate="%{y:.2f} kHz<extra></extra>")
            fig.update_layout(height=300, xaxis_title="", yaxis_title="a.c. Stark shift (kHz)", title=dict(text=f"<b>{title}</b>", font=dict(color='black')), showlegend=False)
            return fig

        # Plot 1
        values2 = np.array([
            self.sim.δ02_bank[0],
            self.sim.δ12_bank[0],
            self.sim.δ22_bank[0],
            self.sim.δ32_bank[0],
            self.sim.δ42_bank[0],
            self.sim.δ52_bank[0]])/(2*np.pi*1e3)

        # Plot 2
        values4 = np.array([
            self.sim.δ04_bank[0],
            self.sim.δ14_bank[0],
            self.sim.δ24_bank[0],
            self.sim.δ34_bank[0],
            self.sim.δ44_bank[0],
            self.sim.δ54_bank[0]])/(2*np.pi*1e3)

        # Plot 3
        values6 = np.array([
            self.sim.δ06_bank[0],
            self.sim.δ16_bank[0],
            self.sim.δ26_bank[0],
            self.sim.δ36_bank[0],
            self.sim.δ46_bank[0],
            self.sim.δ56_bank[0]])/(2*np.pi*1e3)

        # Plot 4
        values8 = np.array([
            self.sim.δ08_bank[0],
            self.sim.δ18_bank[0],
            self.sim.δ28_bank[0],
            self.sim.δ38_bank[0],
            self.sim.δ48_bank[0],
            self.sim.δ58_bank[0]])/(2*np.pi*1e3)

        # 2 x 2 grid
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(make_fig(values2, "2nd order", colors[2]))

        with col2:
            st.plotly_chart(make_fig(values4, "4th order", colors[4]))

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(make_fig(values6, "6th order", colors[6]))

        with col2:
            st.plotly_chart(make_fig(values8, "8th order", colors[8]))

    def plot_zeeman_levels(self):
        m = np.array([-5/2, -3/2, -1/2, 1/2, 3/2, 5/2])
        display_labels = ["⁻⁵⁄₂", "⁻³⁄₂", "⁻¹⁄₂", "⁺¹⁄₂", "⁺³⁄₂", "⁺⁵⁄₂"]
        state_colors = ["#19D3F3", "#FFA15A", "#AB63FA", "#00CC96", "#EF553B", "#636EFA"]
        δ0 = self.sim.δ02_bank[0] + self.sim.δ04_bank[0] + self.sim.δ06_bank[0] + self.sim.δ08_bank[0]
        δ1 = self.sim.δ12_bank[0] + self.sim.δ14_bank[0] + self.sim.δ16_bank[0] + self.sim.δ18_bank[0]
        δ2 = self.sim.δ22_bank[0] + self.sim.δ24_bank[0] + self.sim.δ26_bank[0] + self.sim.δ28_bank[0]
        δ3 = self.sim.δ32_bank[0] + self.sim.δ34_bank[0] + self.sim.δ36_bank[0] + self.sim.δ38_bank[0]
        δ4 = self.sim.δ42_bank[0] + self.sim.δ44_bank[0] + self.sim.δ46_bank[0] + self.sim.δ48_bank[0]
        δ5 = self.sim.δ52_bank[0] + self.sim.δ54_bank[0] + self.sim.δ56_bank[0] + self.sim.δ58_bank[0]
        
        ω01 = (self.sim.ω0 + δ0 - δ1)/(2*np.pi*1e6)
        ω12 = (self.sim.ω0 + δ1 - δ2)/(2*np.pi*1e6)
        ω23 = (self.sim.ω0 + δ2 - δ3)/(2*np.pi*1e6)
        ω34 = (self.sim.ω0 + δ3 - δ4)/(2*np.pi*1e6)
        ω45 = (self.sim.ω0 + δ4 - δ5)/(2*np.pi*1e6)
        
        spacing = np.array([ω45, ω34, ω23, ω12, ω01])  # MHz, placeholders
        y = np.r_[0, np.cumsum(spacing)]
        y = (y-y.min())/(y.max()-y.min())  # rescales levels from 0 → 1

        fig = go.Figure()

        # Zeeman levels
        for mi, yi, label, color in zip(m, y, display_labels, state_colors):
            fig.add_shape(type="line", x0=mi-.35, x1=mi+.35, y0=yi, y1=yi,
                          line=dict(color="black", width=3), layer="below")
            fig.add_trace(go.Scatter(
                x=[mi], y=[yi], mode="markers+text",
                marker=dict(size=55, color=color, opacity=1, line=dict(color="black", width=2)),
                text=[label], textposition="middle center",
                textfont=dict(size=25, color="black"),
                hoverinfo="skip", showlegend=False
            ))

        # Splitting arrows
        for i in range(5):
            xmid = (m[i] + m[i+1])/2
            fig.add_annotation(
                x=xmid, y=y[i+1],
                ax=xmid, ay=(y[i]+y[i+1])/2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="gray"
            )
            fig.add_annotation(
                x=xmid, y=y[i],
                ax=xmid, ay=(y[i]+y[i+1])/2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="gray"
            )
            fig.add_annotation(
                x=xmid, y=(y[i]+y[i+1])/2,
                text=f"{spacing[i]:.2f} MHz",
                showarrow=False, xshift=0
            )

        fig.update_layout(
            height=500,
            xaxis=dict(visible=False, range=[-3,3]),
            yaxis=dict(visible=False, range=[-0.1, 1.1]),
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(l=20,r=20,t=20,b=20)
        )

        st.plotly_chart(fig, use_container_width=True)        
            
    def plot_rabi_frequencies(self):
        labels = ["+5/2", "+3/2", "-1/2", "-5/2", "-3/2", "+1/2"]
        display_labels = ["⁺⁵⁄₂", "⁺³⁄₂", "⁻¹⁄₂", "⁻⁵⁄₂", "⁻³⁄₂", "⁺¹⁄₂"]
        mF = np.array([2.5,1.5,-.5,-2.5,-1.5,.5])
        
        state_colors = {
            "+5/2": "#636EFA",
            "+3/2": "#EF553B",
            "+1/2": "#00CC96",
            "-1/2": "#AB63FA",
            "-3/2": "#FFA15A",
            "-5/2": "#19D3F3"
        }
        
        x0 = -.7
        y0 = .65
        r = .7

        angles = np.pi/2 - np.arange(6)*np.pi/3
        x = x0 + r*np.cos(angles)
        y = y0 + r*np.sin(angles)

        # rabi = st.session_state.get("rabi",np.zeros((6,6)))
        rabi = st.session_state.get("rabi",{})

        st.session_state.setdefault("rabi_transition_v3",(0,2))
        selected = st.session_state.rabi_transition_v3

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            display_mode = st.segmented_control(
                "",
                ["Rabi frequency (kHz)", "π time (μs)"],
                default="Rabi frequency (kHz)",
                label_visibility="collapsed"
            )

        calculate = col2.button("Calculate")
        simulate = col3.button("Simulate")

        pi_time = display_mode == "π time (μs)"

        fig = go.Figure()
        paragraph = """
        Welcome to the high-order Raman calculator.<br>
        Colored circles represent the six Zeeman <br>
        states of the D<sub>5/2</sub> manifold.<br>
        <br>
        Press "Calculate" after selecting desired <br>
        beam parameters in the left tab to calculate <br>
        corresponding 2- 4- and 6-photon rabi <br>
        frequencies. Analytic values are shown<br>
        along lines connecting Zeeman states. <br>
        <br>
        To calculate numerical solutions for a given <br>
        transition, click the line corresponding <br>
        to that transition, and click "Simulate."
        """
        
        fig.add_annotation(
            x=0.8, y=0.65,
            text=paragraph,
            showarrow=False,
            font=dict(size=16),
            align="left"
        )
        
        transitions = [(i,j) for i in range(6) for j in range(i+1,6)]
        order = lambda ij: 2 if abs(mF[ij[0]]-mF[ij[1]])<=2 else 4 if abs(mF[ij[0]]-mF[ij[1]])==3 else 6
        transitions.sort(key=order)

        # max_rabi = max(abs(rabi[i,j]) for i,j in transitions)
        max_rabi = max(map(abs,rabi.values()),default=0)

        def intersection_t(i,j,k,l):
            p,q = np.array([x[i],y[i]]),np.array([x[k],y[k]])
            r,s = np.array([x[j]-x[i],y[j]-y[i]]),np.array([x[l]-x[k],y[l]-y[k]])
            cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
            den = cross(r,s)

            if abs(den)<1e-10:
                return None

            t,u = cross(q-p,s)/den,cross(q-p,r)/den
            return t if 1e-6<t<1-1e-6 and 1e-6<u<1-1e-6 else None

        label_positions = {}

        for i,j in transitions:
            ints = [intersection_t(i,j,k,l) for k,l in transitions if (i,j)!=(k,l)]
            ints = np.sort(np.unique(np.round([t for t in ints if t is not None],8)))

            if len(ints)>=2:
                n = np.argmax(np.diff(ints)); t = (ints[n]+ints[n+1])/2
            elif len(ints)==1:
                t = ints[0]/2 if ints[0]>.5 else (ints[0]+1)/2
            else:
                t = .5

            label_positions[i,j] = t

        hover_traces = []

        for i,j in transitions:
            # value,dx,dy = rabi[i,j],x[j]-x[i],y[j]-y[i]
            key1 = f"{labels[i]}<->{labels[j]}"
            key2 = f"{labels[j]}<->{labels[i]}"
            value = rabi.get(key1,rabi.get(key2,0))

            display_value = np.inf if pi_time and value == 0 else 1e3/(2*abs(value)) if pi_time else value
            display_text = "∞" if np.isinf(display_value) else np.format_float_positional(display_value, precision=3, unique=False, fractional=False, trim='k')

            dx,dy = x[j]-x[i],y[j]-y[i]
            
            photon_order = order((i,j))
            rgb = "0,0,0" if photon_order==2 else "135,206,250" if photon_order==4 else "240,128,128"
            opacity = 1 if (i,j)==selected else .25
            width = max(1, 4 + 2*np.log10(abs(value))) if value else 1
            
            t = label_positions[i,j]
            gap = .015*len(f"{value:.2f}") / np.hypot(dx,dy)
            t1, t2 = max(0,t-gap), min(1,t+gap)

            color = f"rgb({rgb})"

            fig.add_trace(go.Scatter(x=[x[i],x[i]+t1*dx], y=[y[i],y[i]+t1*dy], mode="lines+markers",
                line=dict(color=color,width=width), marker=dict(size=[0,width],color=color),
                opacity=opacity, hoverinfo="skip",showlegend=False))

            fig.add_trace(go.Scatter(x=[x[i]+t2*dx,x[j]], y=[y[i]+t2*dy,y[j]], mode="lines+markers",
                line=dict(color=color,width=width), marker=dict(size=[width,0],color=color),
                opacity=opacity, hoverinfo="skip",showlegend=False))
   
            lx,ly = x[i]+t*dx,y[i]+t*dy
            
            # fig.add_shape(type="circle", x0=lx-.015*len(f"{value:.2f}"), x1=lx+.015*len(f"{value:.2f}"), y0=ly-.055, y1=ly+.055, fillcolor="white", line_width=0)
            fig.add_annotation(x=lx,y=ly,text=display_text,showarrow=False,font=dict(size=11,color="black"))
            tc = np.linspace(.08,.92,30)

            hover_traces.append(go.Scatter(
                x=x[i]+tc*dx,y=y[i]+tc*dy,mode="markers",
                marker=dict(size=15,color="rgba(0,0,0,0.001)"),
                customdata=[[i,j]]*len(tc),
                hovertemplate=f"{labels[i]} ↔ {labels[j]}<br>{photon_order}-photon<br>Ω = {value:.2f} kHz<extra></extra>",
                selected=dict(marker=dict(opacity=0)),
                unselected=dict(marker=dict(opacity=0)),
                showlegend=False))

        for trace in hover_traces:
            fig.add_trace(trace)

        borders = ["black" if i in selected else "rgba(0,0,0,.25)" for i in range(6)]
        texts = ["black" if i in selected else "rgba(0,0,0,.25)" for i in range(6)]

        fig.add_trace(go.Scatter(
            x=x,y=y,mode="markers+text",
            marker=dict(size=65,color=[state_colors[label] for label in labels],line=dict(color=borders,width=2)),
            text=display_labels,textposition="middle center",textfont=dict(size=32,color="black"),
            hoverinfo="skip",showlegend=False))

        for name,color in [("2-photon","black"),("4-photon","lightskyblue"),("6-photon","lightcoral")]:
            fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",line=dict(color=color,width=4), name=name,hoverinfo="skip",showlegend=True))

        fig.update_layout(
            height=450, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(visible=False, range=[-1.35,1.35], scaleanchor="y", scaleratio=1, fixedrange=True),
            yaxis=dict(visible=False, range=[-0.2,1.5], fixedrange=True),
            dragmode=False, plot_bgcolor="white", clickmode="event+select", showlegend=True,
            legend=dict(x=.0,y=1,xanchor="left",yanchor="top", bgcolor="rgba(255,255,255,.8)"))
        event = st.plotly_chart(fig,use_container_width=True,key="rabi_diagram_v3", on_select="rerun",selection_mode="points")

        if event and event.selection.points:
            data = event.selection.points[-1].get("customdata")

            if data is not None:
                new_selected = tuple(map(int,data))

                if new_selected != selected:
                    st.session_state.rabi_transition_v3 = new_selected
                    st.rerun()

        i,j = st.session_state.rabi_transition_v3

        state_order = {
            "+5/2": 0, "+3/2": 1, "+1/2": 2,
            "-1/2": 3, "-3/2": 4, "-5/2": 5
        }

        a,b = labels[i],labels[j]
        self.sim.transition = f"{a}<->{b}" if state_order[a] < state_order[b] else f"{b}<->{a}"
        return calculate, simulate
    
    def save_simulation(self):
        st.session_state.simulation = {
            "transition": self.sim.transition,
            "ω_bank": self.sim.ω_bank,
            "fit_res": self.sim.fit_res,
            "fit": self.sim.fit,
            "p52_res": self.sim.p52_res,
            "p32_res": self.sim.p32_res,
            "p12_res": self.sim.p12_res,
            "m12_res": self.sim.m12_res,
            "m32_res": self.sim.m32_res,
            "m52_res": self.sim.m52_res,
            "t": self.sim.t,
            "p52": self.sim.p52,
            "p32": self.sim.p32,
            "p12": self.sim.p12,
            "m12": self.sim.m12,
            "m32": self.sim.m32,
            "m52": self.sim.m52,
            'δ2': self.sim.δ2,
            'δ4': self.sim.δ4,
            'δ6': self.sim.δ6,
            'δ8': self.sim.δ8
        }

    def plot_simulation(self):
        d = st.session_state.simulation

        labels = ["+5/2","+3/2","+1/2","-1/2","-3/2","-5/2"]
        res = ["p52_res","p32_res","p12_res","m12_res","m32_res","m52_res"]
        flop = ["p52","p32","p12","m12","m32","m52"]

        fig1, fig2 = go.Figure(), go.Figure()

        colors = {
            "+5/2": "#636EFA", "+3/2": "#EF553B", "+1/2": "#00CC96",
            "-1/2": "#AB63FA", "-3/2": "#FFA15A", "-5/2": "#19D3F3"
        }

        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["p52_res"], mode="markers", name="+5/2", marker_color=colors["+5/2"])
        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["p32_res"], mode="markers", name="+3/2", marker_color=colors["+3/2"])
        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["p12_res"], mode="markers", name="+1/2", marker_color=colors["+1/2"])
        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["m12_res"], mode="markers", name="-1/2", marker_color=colors["-1/2"])
        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["m32_res"], mode="markers", name="-3/2", marker_color=colors["-3/2"])
        fig1.add_scatter(x=d["ω_bank"]/(2*np.pi*1e6), y=d["m52_res"], mode="markers", name="-5/2", marker_color=colors["-5/2"])
        fig1.add_scatter(x=d["fit_res"][0]/(2*np.pi*1e6), y=d["fit_res"][1], mode="lines", line=dict(color="black"))

        fig2.add_scatter(x=d["t"]*1e6, y=d["p52"], mode="markers", name="+5/2", marker_color=colors["+5/2"])
        fig2.add_scatter(x=d["t"]*1e6, y=d["p32"], mode="markers", name="+3/2", marker_color=colors["+3/2"])
        fig2.add_scatter(x=d["t"]*1e6, y=d["p12"], mode="markers", name="+1/2", marker_color=colors["+1/2"])
        fig2.add_scatter(x=d["t"]*1e6, y=d["m12"], mode="markers", name="-1/2", marker_color=colors["-1/2"])
        fig2.add_scatter(x=d["t"]*1e6, y=d["m32"], mode="markers", name="-3/2", marker_color=colors["-3/2"])
        fig2.add_scatter(x=d["t"]*1e6, y=d["m52"], mode="markers", name="-5/2", marker_color=colors["-5/2"])
        fig2.add_scatter(x=d["fit"][0]*1e6, y=d["fit"][1], mode="lines", line=dict(color="black"), showlegend = False)

        fig1.update_layout(title = 'spectroscopy', height=300, showlegend=False, margin=dict(t=50), xaxis_title="beam detuning (MHz)", yaxis_title="Population", yaxis_range=[-.05,1.05])
        fig2.update_layout(title = 'resonant dynamics', height=300, margin=dict(t=50), xaxis_title="time (μs)", yaxis_title="Population", yaxis_range=[-.05,1.05])
        # Maximum population outside the driven transition
        driven_states = d["transition"].split("<->")

        population_data = {
            "+5/2": d["p52"],
            "+3/2": d["p32"],
            "+1/2": d["p12"],
            "-1/2": d["m12"],
            "-3/2": d["m32"],
            "-5/2": d["m52"]
        }

        max_pop = [
            None if s in driven_states else np.max(population_data[s])
            for s in labels
        ]

        fig3 = go.Figure()
        fig3.add_bar(
            x=labels,
            y=max_pop,
            marker_color=[colors[s] for s in labels],
            text=[None if x is None else f"{x:.2g}" for x in max_pop],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y:.4f}<extra></extra>"
        )

        fig3.update_layout(
            title="Maximum off-resonant population",
            height=300,
            margin=dict(t=50),
            xaxis_title="",
            yaxis_title="Maximum population",
            yaxis_range=[0, max(x for x in max_pop if x is not None)*1.1],
            showlegend=False
        )
        
        # Stark shift contributions
        stark_orders = ["2nd order", "4th order", "6th order", "8th order"]
        stark_percentages = [70, 20, 8, 2]   # placeholder percentages
        stark_shifts = [d["δ2"]/(2*np.pi*1e3), d["δ4"]/(2*np.pi*1e3), d["δ6"]/(2*np.pi*1e3), d["δ8"]/(2*np.pi*1e3)]
        total = abs(d["δ2"]) + abs(d["δ4"]) + abs(d["δ6"]) + abs(d["δ8"])
        stark_percentages = [100*abs(d["δ2"])/total, 100*abs(d["δ4"])/total, 100*abs(d["δ6"])/total, 100*abs(d["δ8"])/total]
                             
                             
        stark_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

        # Pie chart
        fig4 = go.Figure(data=[go.Pie(
            labels=stark_orders,
            values=stark_percentages,
            marker=dict(colors=stark_colors),
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:.2f}%<extra></extra>"
        )])

        fig4.update_layout(
            height=260,
            margin=dict(t=30, l=5, r=5, b=5),
            showlegend=False
        )

        # Stacked Stark-shift bar
        fig5 = go.Figure()

        for order, shift, color in zip(stark_orders, stark_shifts, stark_colors):
            fig5.add_bar(
                x=[""],
                y=[shift],
                name=order,
                marker_color=color,
                hovertemplate=f"{order}<br>{shift} kHz<extra></extra>"
            )
            
        fig5.update_layout(
            title="Stark shift contributions",
            barmode="stack",
            height=260,
            margin=dict(t=50, l=40, r=0, b=20),
            xaxis=dict(
                domain=[0.00, 0.25],
                showticklabels=False,
                title=""
            ),
            yaxis=dict(title="Stark shift (kHz)"),
            showlegend=True,
            legend=dict(
                x=0.50,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                orientation="v"
            )
        )
        
        a, b = d["transition"].split("<->")

        display = {
            "+5/2":"⁺⁵⁄₂", "+3/2":"⁺³⁄₂", "+1/2":"⁺¹⁄₂",
            "-1/2":"⁻¹⁄₂", "-3/2":"⁻³⁄₂", "-5/2":"⁻⁵⁄₂"
        }

        def sphere(state):
            return f'<span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:50%;background:{colors[state]};border:2px solid black;font-size:19px;font-weight:500;color:black;">{display[state]}</span>'

        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:left;gap:0px;width:100%;">'
            f'{sphere(a)}'
            f'<span style="display:inline-block;width:70px;text-align:center;font-size:26px;">⟷</span>'
            f'{sphere(b)}'
            f'</div>',
            unsafe_allow_html=True
        )

        col1,col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1,use_container_width=True)
            st.plotly_chart(fig3,use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)

            bar_col, pie_col = st.columns([2, 1], gap=None)

            with bar_col:
                st.plotly_chart(fig5, use_container_width=True)

            with pie_col:
                st.plotly_chart(fig4, use_container_width=True)
                
if __name__ == "__main__":
    App()