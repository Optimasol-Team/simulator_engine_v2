from datetime import date, datetime, time as dt_time, timedelta
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from optimiser_engine.domain import Client
from optimiser_engine.engine.models.Exceptions import SolverFailed, WeatherNotValid
from optimiser_engine.engine.models.trajectory import RouterMode, StandardWHType, TrajectorySystem
from optimiser_engine.engine.service import OptimizerService


APP_DIR = Path(__file__).resolve().parent


def load_example_files() -> Tuple[str, str]:
    """Read bundled YAML/CSV examples to prefill the UI."""
    yaml_path = APP_DIR / "client_sample.yaml"
    csv_path = APP_DIR / "weather.csv"

    yaml_text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    csv_text = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    return yaml_text, csv_text


def init_session_defaults(default_yaml: str, default_csv: str) -> None:
    """Seed session_state with defaults the first time the app loads."""
    defaults = {
        "client_text": default_yaml,
        "solar_text": default_csv,
        "client": None,
        "client_ok": False,
        "client_errors": None,
        "solar_df": None,
        "solar_ok": False,
        "solar_errors": None,
        "opt_result": None,
        "opt_error": None,
        "std_result": None,
        "router_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_client_from_yaml(yaml_text: str) -> Tuple[Optional[Client], Optional[str]]:
    """Parse client configuration YAML and return a Client or an error string."""
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return None, "Le YAML doit décrire un objet (mapping)."
        client = Client.from_dict(data)
        return client, None
    except Exception as exc:  # Broad on purpose to surface parsing issues
        return None, f"Erreur dans la configuration client : {exc}"


def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of a datetime column."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    for col in df.columns:
        name = str(col).lower()
        if "date" in name or "time" in name:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().any():
                df[col] = parsed
                return col

    # Fallback: try the first column
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col], errors="coerce", utc=False, infer_datetime_format=True)
    if parsed.notna().any():
        df[first_col] = parsed
        return first_col
    return None


def parse_solar_from_csv(csv_text: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Parse CSV text into a DataFrame with a datetime index and one production column."""
    try:
        df = pd.read_csv(StringIO(csv_text), sep=None, engine="python")
    except Exception as exc:
        return None, f"Erreur de lecture du CSV : {exc}"

    if df.empty:
        return None, "Le CSV est vide."

    datetime_col = _detect_datetime_column(df)
    if datetime_col is None:
        return None, "Aucune colonne de date/heure trouvée ou parsable."

    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None, "Aucune colonne numérique détectée pour la production solaire."

    production_col = numeric_cols[0]
    clean_df = df[[production_col]].copy()
    clean_df = clean_df.rename(columns={production_col: "production"})
    clean_df.index = pd.to_datetime(clean_df.index, errors="coerce")
    clean_df = clean_df[clean_df.index.notna()]

    if clean_df.empty:
        return None, "Le CSV ne contient pas de lignes valides après parsing."

    return clean_df, None


def _build_time_axes(traj: TrajectorySystem) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Create aligned time axes for decisions (N) and temperatures (N+1)."""
    ctx = traj.context
    base_index = pd.date_range(
        start=ctx.reference_datetime,
        periods=ctx.N,
        freq=f"{int(ctx.step_minutes)}T",
    )
    temp_index = pd.date_range(
        start=ctx.reference_datetime,
        periods=ctx.N + 1,
        freq=f"{int(ctx.step_minutes)}T",
    )
    return base_index, temp_index


def extract_series(traj: TrajectorySystem) -> Dict[str, pd.Series]:
    """Convert a trajectory into plot-ready pandas Series."""
    base_index, temp_index = _build_time_axes(traj)

    temps = traj.get_temperatures()
    decisions = traj.get_decisions()
    imports = traj.get_imports()
    exports = traj.get_exports()
    ctx = traj.context
    power = traj.config_system.power if traj.config_system else None

    series: Dict[str, pd.Series] = {}
    if temps is not None:
        series["temperature"] = pd.Series(temps, index=temp_index, name="Température (°C)")
    if decisions is not None:
        series["x"] = pd.Series(decisions, index=base_index, name="x (0-1)")
    if decisions is not None and power is not None:
        series["power_kw"] = pd.Series(decisions * power / 1000.0, index=base_index, name="Puissance CE (kW)")
    if ctx and ctx.solar_production is not None:
        series["pv_kw"] = pd.Series(ctx.solar_production / 1000.0, index=base_index, name="PV (kW)")
    if imports is not None:
        series["imports_kw"] = pd.Series(imports / 1000.0, index=base_index, name="Import (kW)")
    if exports is not None:
        series["exports_kw"] = pd.Series(exports / 1000.0, index=base_index, name="Export (kW)")
    return series


def run_optimized(client: Client, solar_df: pd.DataFrame, params: Dict) -> Dict:
    """Run the optimiser and return trajectory plus KPIs."""
    service = OptimizerService(params["horizon_hours"], params["step_minutes"])
    start_dt = params["start_datetime"]
    initial_temp = params["initial_temperature"]

    t0 = perf_counter()
    traj = service.trajectory_of_client(client, start_dt, initial_temp, solar_df)
    duration = perf_counter() - t0

    cost = traj.compute_cost()
    autocons = traj.compute_self_consumption()
    return {
        "trajectory": traj,
        "cost": cost,
        "autocons": autocons,
        "duration": duration,
        "series": extract_series(traj),
    }


def run_baseline(
    client: Client,
    solar_df: pd.DataFrame,
    params: Dict,
    mode,
    kind: str,
) -> Dict:
    """Run standard or router baseline according to kind."""
    service = OptimizerService(params["horizon_hours"], params["step_minutes"])
    start_dt = params["start_datetime"]
    initial_temp = params["initial_temperature"]
    setpoint_temp = params.get("baseline_setpoint")

    t0 = perf_counter()
    if kind == "standard":
        traj = service.trajectory_of_client_standard(
            client,
            start_dt,
            initial_temp,
            solar_df,
            mode_WH=mode,
            setpoint_temperature=setpoint_temp,
        )
        # Compute temperatures/import/export before KPIs
        traj.update_X()
    else:
        traj = service.trajectory_of_client_router(
            client,
            start_dt,
            initial_temp,
            solar_df,
            router_mode=mode,
            setpoint_temperature=setpoint_temp,
        )
    duration = perf_counter() - t0

    cost = traj.compute_cost()
    autocons = traj.compute_self_consumption()
    return {
        "trajectory": traj,
        "cost": cost,
        "autocons": autocons,
        "duration": duration,
        "series": extract_series(traj),
    }


def _plot_single_mode(label: str, series: Dict[str, pd.Series]) -> None:
    """Plot x, T, E, I for one mode using matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    fig.suptitle(label)

    ax_x, ax_t = axes[0]
    ax_i, ax_e = axes[1]

    if "x" in series:
        ax_x.plot(series["x"].index, series["x"].values, label="x")
    ax_x.set_title("x (0-1)")
    ax_x.grid(True)

    if "temperature" in series:
        ax_t.plot(series["temperature"].index, series["temperature"].values, label="T (°C)", color="tab:orange")
    ax_t.set_title("Température (°C)")
    ax_t.grid(True)

    if "imports_kw" in series:
        ax_i.plot(series["imports_kw"].index, series["imports_kw"].values, label="I (kW)", color="tab:green")
    ax_i.set_title("Import (kW)")
    ax_i.grid(True)

    if "exports_kw" in series:
        ax_e.plot(series["exports_kw"].index, series["exports_kw"].values, label="E (kW)", color="tab:red")
    ax_e.set_title("Export (kW)")
    ax_e.grid(True)

    def _format_time_axis(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        formatter = mdates.DateFormatter("%m-%d %H:%M")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", rotation=20)

    for ax in (ax_x, ax_t, ax_i, ax_e):
        if ax.get_lines():
            ax.legend()
        _format_time_axis(ax)

    st.pyplot(fig, clear_figure=True)


def plot_results(results: Dict[str, Dict]) -> None:
    """Plot separate figures per mode."""
    if not results:
        st.info("Aucune trajectoire à afficher pour le moment.")
        return
    for label, res in results.items():
        series = res.get("series", {})
        if series:
            st.subheader(f"Courbes {label}")
            _plot_single_mode(label, series)


def display_kpis(label: str, result: Dict) -> None:
    """Render KPI metrics for a single run."""
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Coût {label}", f"{result['cost']:.2f}")
    col2.metric(f"Temps {label}", f"{result['duration']:.2f} s")
    col3.metric(f"Autoconsommation {label}", f"{result['autocons'] * 100:.1f} %")


def display_summary_table(results: Dict[str, Dict]) -> None:
    """Display aggregated cost/autocons in a table."""
    if not results:
        return
    rows = []
    for label, res in results.items():
        rows.append(
            {
                "Mode": label,
                "Coût": round(res["cost"], 2),
                "Autoconsommation (%)": round(res["autocons"] * 100, 1),
            }
        )
    table_df = pd.DataFrame(rows).set_index("Mode")
    st.table(table_df)


def plot_pv_data(solar_df: Optional[pd.DataFrame]) -> None:
    """Plot PV production from the input data."""
    if solar_df is None or solar_df.empty:
        return
    pv_series = solar_df.iloc[:, 0]
    pv_series.name = pv_series.name or "PV"
    st.subheader("Production PV (données)")
    st.line_chart(pv_series)


def main() -> None:
    st.set_page_config(page_title="Simulateur Optimisation CE", layout="wide")
    sample_yaml, sample_csv = load_example_files()
    init_session_defaults(sample_yaml, sample_csv)

    st.title("Simulateur d'optimisation chauffe-eau")
    st.caption("Chargez votre configuration, validez les données, puis lancez l'optimisation et les baselines.")

    col_yaml, col_csv = st.columns(2)
    with col_yaml:
        st.subheader("Configuration client (YAML)")
        client_text = st.text_area(
            "Edition YAML",
            key="client_text",
            height=420,
        )
        if st.button("Valider le client"):
            client, err = parse_client_from_yaml(client_text)
            if err:
                st.session_state.client_ok = False
                st.session_state.client_errors = err
                st.error(err)
            else:
                st.session_state.client = client
                st.session_state.client_ok = True
                st.session_state.client_errors = None
                st.success("Client chargé avec succès.")
        if st.session_state.client_errors:
            st.error(st.session_state.client_errors)

    with col_csv:
        st.subheader("Production solaire (CSV)")
        solar_text = st.text_area(
            "Edition CSV",
            key="solar_text",
            height=420,
        )
        if st.button("Valider la production solaire"):
            df, err = parse_solar_from_csv(solar_text)
            if err:
                st.session_state.solar_ok = False
                st.session_state.solar_errors = err
                st.error(err)
            else:
                st.session_state.solar_df = df
                st.session_state.solar_ok = True
                st.session_state.solar_errors = None
                st.success("Production solaire chargée.")
        if st.session_state.solar_errors:
            st.error(st.session_state.solar_errors)

    st.markdown("---")
    st.subheader("Paramètres de simulation")
    ready = st.session_state.client_ok and st.session_state.solar_ok

    default_start = datetime(2026, 1, 1, 0, 0)
    col_params = st.columns(4)
    with col_params[0]:
        horizon_hours = int(
            st.number_input("Horizon (heures)", min_value=1, max_value=48, value=24, step=1)
        )
    with col_params[1]:
        max_step = max(5, horizon_hours * 30)
        step_minutes = int(
            st.number_input("Pas (minutes)", min_value=5, max_value=max_step, value=15, step=5)
        )
    with col_params[2]:
        initial_temperature = float(
            st.number_input("Température initiale (°C)", min_value=5.0, max_value=99.0, value=45.0, step=1.0)
        )
    with col_params[3]:
        baseline_setpoint = float(
            st.number_input("Consigne thermostat (°C)", min_value=30.0, max_value=99.0, value=60.0, step=1.0)
        )

    col_dt = st.columns(2)
    with col_dt[0]:
        start_date = st.date_input("Date de début", value=default_start.date())
    with col_dt[1]:
        start_time = st.time_input("Heure de début", value=default_start.time())

    start_datetime = datetime.combine(start_date, start_time)
    params = {
        "horizon_hours": horizon_hours,
        "step_minutes": step_minutes,
        "start_datetime": start_datetime,
        "initial_temperature": initial_temperature,
        "baseline_setpoint": baseline_setpoint,
    }

    st.markdown("---")
    st.subheader("Optimisation")
    if st.button("Lancer la simulation optimisée", disabled=not ready):
        if not ready:
            st.warning("Chargez et validez d'abord le client et la production solaire.")
        else:
            with st.spinner("Optimisation en cours..."):
                try:
                    st.session_state.opt_result = run_optimized(
                        st.session_state.client, st.session_state.solar_df, params
                    )
                    st.session_state.opt_error = None
                except (WeatherNotValid, SolverFailed) as exc:
                    st.session_state.opt_result = None
                    st.session_state.opt_error = str(exc)
                    st.error(str(exc))
                except Exception as exc:
                    st.session_state.opt_result = None
                    st.session_state.opt_error = str(exc)
                    st.error(f"Echec de l'optimisation : {exc}")

    if st.session_state.opt_error:
        st.error(st.session_state.opt_error)

    st.markdown("---")
    st.subheader("Comparaison standard / routeur")
    standard_options = {
        "Consigne": StandardWHType.SETPOINT,
        "Consigne HC": StandardWHType.SETPOINT_OFF_PEAK,
    }
    router_options = {
        "Autoconsommation pure": RouterMode.SELF_CONSUMPTION_ONLY,
        "Confort (appoint HC)": RouterMode.COMFORT,
    }

    col_modes = st.columns(2)
    with col_modes[0]:
        chosen_standard = st.selectbox(
            "Mode standard",
            options=list(standard_options.keys()),
            index=0,
            disabled=not ready,
        )
    with col_modes[1]:
        chosen_router = st.selectbox(
            "Mode routeur",
            options=list(router_options.keys()),
            index=1 if len(router_options) > 1 else 0,
            disabled=not ready,
        )

    if st.button("Comparer", disabled=not ready):
        if not ready:
            st.warning("Chargez et validez d'abord le client et la production solaire.")
        else:
            with st.spinner("Calcul des baselines..."):
                try:
                    st.session_state.std_result = run_baseline(
                        st.session_state.client,
                        st.session_state.solar_df,
                        params,
                        standard_options[chosen_standard],
                        "standard",
                    )
                    st.session_state.router_result = run_baseline(
                        st.session_state.client,
                        st.session_state.solar_df,
                        params,
                        router_options[chosen_router],
                        "router",
                    )
                except WeatherNotValid as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Echec d'une simulation baseline : {exc}")

    results_to_plot: Dict[str, Dict] = {}
    if st.session_state.opt_result:
        results_to_plot["Optimisé"] = st.session_state.opt_result
    if st.session_state.std_result:
        results_to_plot["Standard"] = st.session_state.std_result
    if st.session_state.router_result:
        results_to_plot["Routeur"] = st.session_state.router_result

    if results_to_plot:
        st.markdown("---")
        st.subheader("Tableau récapitulatif coûts / autoconsommation")
        display_summary_table(results_to_plot)

    st.markdown("---")
    st.subheader("Courbes")
    plot_results(results_to_plot)
    plot_pv_data(st.session_state.solar_df)


if __name__ == "__main__":
    main()
