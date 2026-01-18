from datetime import datetime, time as dt_time, date
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
from matplotlib.gridspec import GridSpec
from optimiser_engine.domain import Client
from optimiser_engine.engine.models.Exceptions import SolverFailed, WeatherNotValid
from optimiser_engine.engine.models.trajectory import RouterMode, StandardWHType, TrajectorySystem
from optimiser_engine.engine.service import OptimizerService

APP_DIR = Path(__file__).resolve().parent

def load_example_files() -> Tuple[str, str]:
    yaml_path = APP_DIR / "client_sample.yaml"
    csv_path = APP_DIR / "weather.csv"
    yaml_text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    csv_text = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    return yaml_text, csv_text

def init_session_defaults(default_yaml: str, default_csv: str) -> None:
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
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return None, "Le YAML doit décrire un objet (mapping)."
        client = Client.from_dict(data)
        return client, None
    except Exception as exc:
        return None, f"Erreur dans la configuration client : {exc}"

def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
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
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col], errors="coerce", utc=False, infer_datetime_format=True)
    if parsed.notna().any():
        df[first_col] = parsed
        return first_col
    return None

def parse_solar_from_csv(csv_text: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
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

def _plot_comparison(results: Dict[str, Dict]) -> None:
    if not results:
        st.info("Aucune trajectoire à afficher pour le moment.")
        return
    mode_colors = {
        "Sans routeur": "#1f77b4",
        "Avec routeur": "#ff7f0e",
        "Optimisé": "#2ca02c",
    }
    all_series = {}
    for mode, res in results.items():
        if "series" in res:
            all_series[mode] = res["series"]
    if not all_series:
        return
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax_temp = fig.add_subplot(gs[0])
    ax_power = fig.add_subplot(gs[1], sharex=ax_temp)
    for mode, series in all_series.items():
        if "temperature" in series:
            color = mode_colors.get(mode, None)
            ax_temp.plot(series["temperature"].index, 
                        series["temperature"].values, 
                        label=mode,
                        color=color,
                        linewidth=2)
    ax_temp.set_ylabel('Température (°C)')
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(loc='upper right')
    ax_temp.set_title('Comparaison des températures')
    for mode, series in all_series.items():
        if "x" in series:
            color = mode_colors.get(mode, None)
            ax_power.plot(series["x"].index, 
                         series["x"].values, 
                         label=mode,
                         color=color,
                         linewidth=2)
    ax_power.set_ylabel('x (0-1)')
    ax_power.set_xlabel('Temps')
    ax_power.grid(True, alpha=0.3)
    ax_power.legend(loc='upper right')
    ax_power.set_title('Comparaison des commandes (x)')
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%H:%M\n%m/%d")
    ax_power.xaxis.set_major_locator(locator)
    ax_power.xaxis.set_major_formatter(formatter)
    ax_power.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    if st.checkbox("Afficher les importations/exportations"):
        st.subheader("Importations et Exportations")
        fig2, (ax_import, ax_export) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for mode, series in all_series.items():
            if "imports_kw" in series:
                color = mode_colors.get(mode, None)
                ax_import.plot(series["imports_kw"].index, 
                             series["imports_kw"].values, 
                             label=mode,
                             color=color,
                             linewidth=2)
        ax_import.set_ylabel('Import (kW)')
        ax_import.grid(True, alpha=0.3)
        ax_import.legend(loc='upper right')
        ax_import.set_title('Importations d\'électricité')
        for mode, series in all_series.items():
            if "exports_kw" in series:
                color = mode_colors.get(mode, None)
                ax_export.plot(series["exports_kw"].index, 
                             series["exports_kw"].values, 
                             label=mode,
                             color=color,
                             linewidth=2)
        ax_export.set_ylabel('Export (kW)')
        ax_export.set_xlabel('Temps')
        ax_export.grid(True, alpha=0.3)
        ax_export.legend(loc='upper right')
        ax_export.set_title('Exportations d\'électricité')
        ax_export.xaxis.set_major_locator(locator)
        ax_export.xaxis.set_major_formatter(formatter)
        ax_export.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        st.pyplot(fig2)

def display_summary_table(results: Dict[str, Dict]) -> None:
    if not results:
        return
    rows = []
    for label, res in results.items():
        rows.append({
            "Mode": label,
            "Coût (€)": f"{res['cost']:.2f}",
            "Autoconsommation (%)": f"{res['autocons'] * 100:.1f}",
            "Temps calcul (s)": f"{res['duration']:.2f}"
        })
    table_df = pd.DataFrame(rows).set_index("Mode")
    st.table(table_df)

def plot_pv_data(solar_df: Optional[pd.DataFrame]) -> None:
    if solar_df is None or solar_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    pv_series = solar_df.iloc[:, 0]
    ax.plot(pv_series.index, pv_series.values, color='#ffd700', linewidth=2)
    ax.set_ylabel('Production PV (kW)')
    ax.set_xlabel('Temps')
    ax.grid(True, alpha=0.3)
    ax.set_title('Production Photovoltaïque')
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%H:%M\n%m/%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def main() -> None:
    st.set_page_config(page_title="Simulateur Optimisation CE", layout="wide")
    sample_yaml, sample_csv = load_example_files()
    init_session_defaults(sample_yaml, sample_csv)
    st.title("Simulateur d'optimisation chauffe-eau")
    st.caption("Chargez votre configuration, validez les données, puis lancez les simulations.")
    col_yaml, col_csv = st.columns(2)
    with col_yaml:
        st.subheader("Configuration client (YAML)")
        client_text = st.text_area(
            "Edition YAML",
            key="client_text",
            height=380,
            help="Configuration du client au format YAML"
        )
        if st.button("Valider le client", key="validate_client"):
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
            height=380,
            help="Données de production solaire au format CSV"
        )
        if st.button("Valider la production solaire", key="validate_solar"):
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
    if not ready:
        st.warning("Veuillez d'abord charger et valider les données client et solaire.")
    default_start = datetime(2026, 1, 1, 0, 0)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        horizon_hours = int(st.number_input("Horizon (heures)", min_value=1, max_value=48, value=24, step=1))
    with col2:
        max_step = max(5, horizon_hours * 30)
        step_minutes = int(st.number_input("Pas (minutes)", min_value=5, max_value=max_step, value=15, step=5))
    with col3:
        initial_temperature = float(st.number_input("Température initiale (°C)", min_value=5.0, max_value=99.0, value=45.0, step=1.0))
    with col4:
        baseline_setpoint = float(st.number_input("Consigne thermostat (°C)", min_value=30.0, max_value=99.0, value=60.0, step=1.0))
    col_date, col_time = st.columns(2)
    with col_date:
        start_date = st.date_input("Date de début", value=default_start.date())
    with col_time:
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
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        if st.button("Lancer toutes les simulations", type="primary", disabled=not ready):
            if not ready:
                st.warning("Veuillez d'abord valider les données client et solaire.")
            else:
                with st.spinner("Calcul en cours..."):
                    try:
                        st.session_state.std_result = run_baseline(
                            st.session_state.client,
                            st.session_state.solar_df,
                            params,
                            StandardWHType.SETPOINT,
                            "standard",
                        )
                        st.session_state.router_result = run_baseline(
                            st.session_state.client,
                            st.session_state.solar_df,
                            params,
                            RouterMode.COMFORT,
                            "router",
                        )
                        st.session_state.opt_result = run_optimized(
                            st.session_state.client, st.session_state.solar_df, params
                        )
                        st.session_state.opt_error = None
                        st.success("Simulations terminées avec succès!")
                    except (WeatherNotValid, SolverFailed) as exc:
                        st.session_state.opt_result = None
                        st.session_state.opt_error = str(exc)
                        st.error(str(exc))
                    except Exception as exc:
                        st.session_state.opt_result = None
                        st.session_state.opt_error = str(exc)
                        st.error(f"Erreur lors des simulations : {exc}")
    with col_sim2:
        if st.button("Réinitialiser les résultats", type="secondary"):
            st.session_state.opt_result = None
            st.session_state.std_result = None
            st.session_state.router_result = None
            st.rerun()
    if st.session_state.opt_error:
        st.error(st.session_state.opt_error)
    results_to_plot: Dict[str, Dict] = {}
    if st.session_state.std_result:
        results_to_plot["Sans routeur"] = st.session_state.std_result
    if st.session_state.router_result:
        results_to_plot["Avec routeur"] = st.session_state.router_result
    if st.session_state.opt_result:
        results_to_plot["Optimisé"] = st.session_state.opt_result
    if results_to_plot:
        st.markdown("---")
        st.subheader("Tableau récapitulatif")
        display_summary_table(results_to_plot)
        st.markdown("---")
        st.subheader("Comparaison visuelle")
        _plot_comparison(results_to_plot)
        st.markdown("---")
        plot_pv_data(st.session_state.solar_df)

if __name__ == "__main__":
    main()