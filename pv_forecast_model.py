# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:43:45 2025

@author: wofracko
"""

import pandas as pd
import pvlib
import getpass
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import sys
import logging
import holidays
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Konfiguracja logowania ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

USER = getpass.getuser()

# --- Import z repozytorium ---
sys.path.append(r"C:\Repos\electricity-consumption_tpa")
sys.path.append(r"C:\Repos\power_pv_forecasts")
#sys.path.append(rf"C:\Users\{USER}\EWE\SP_Dzial_RH - Zasoby\Projekty\SOGL prognozowanie")
from EE_measurements_downloader import run_data_download
import prepare_data


# --- Definicja instalacji ---
INSTALLATIONS = [
    {"name": "Hala_A_246", "ppe": "590322400100343246", "capacity": 155.2, "tilt": 10, "azimuth": 238},
    {"name": "Hala_B_246", "ppe": "590322400100343246", "capacity": 53.4, "tilt": 11, "azimuth": 222},
    {"name": "Hala_D_246", "ppe": "590322400100343246", "capacity": 7.8, "tilt": 15, "azimuth": 148},
    {"name": "Biurowiec_749", "ppe": "590322400101395749", "capacity": 20.475, "tilt": 16, "azimuth": 34},
    {"name": "Wiata_749", "ppe": "590322400101395749", "capacity": 92.0, "tilt": 6, "azimuth": 147},
    {"name": "Hala_A_749", "ppe": "590322400101395749", "capacity": 14.1, "tilt": 11, "azimuth": 73},
    {"name": "Hala_B_749", "ppe": "590322400101395749", "capacity": 159.1, "tilt": 10, "azimuth": 58},
    {"name": "Hala_C_749", "ppe": "590322400101395749", "capacity": 40.3, "tilt": 11, "azimuth": 73},
]

# --- Ustawienia ---

BASE_PATH = Path(rf"C:\Users\{USER}\EWE\SP_Dzial_RH - Zasoby\Projekty\SOGL prognozowanie\\")

WEATHER_CSV = BASE_PATH / "prognoza_Hirschvogel/prognozy_pogody.csv"
SHUTDOWNS_CSV = BASE_PATH / "prognoza_Hirschvogel/wyłączenia_PV.csv"
PV_MAX_POTENTIAL_CSV = BASE_PATH / "prognoza_Hirschvogel/max_potential.csv"
PV_FORECAST_CSV = BASE_PATH / "prognoza_Hirschvogel/prognoza_PV.csv"
SOGL_PATH = BASE_PATH / "pliki_SOGL"
MEASUREMENTS_PATH = BASE_PATH / "prognoza_Hirschvogel/dane_pomiarowe_Hirschvogel.csv"
MEASUREMENTS_FORECAST_PATH = BASE_PATH / "prognoza_Hirschvogel/prognoza_zużycia_Hirschvogel.csv"

TZ = pytz.timezone("Europe/Warsaw")


DURATION_DAYS = 25

# --- Parametry paneli ---
GAMMA = -0.0045  # współczynnik temperaturowy [%/°C]


# --- Shutdowns ---
def load_shutdowns(csv_file: Path) -> list:
    """Ładuje plik CSV z wyłączeniami PV."""
    df = pd.read_csv(csv_file, sep=";", parse_dates=["start", "end"])
    return df.to_dict(orient="records")


def apply_shutdowns(df: pd.DataFrame, shutdowns: list) -> pd.DataFrame:
    """Zeruje produkcję PV w okresach wyłączeń."""
    tz = df.index.tz
    for shutdown in shutdowns:
        if shutdown["name"] not in df.columns:
            continue

        start = pd.Timestamp(shutdown["start"])
        end = pd.Timestamp(shutdown["end"])

        start = start.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT") if start.tzinfo is None else start.tz_convert(tz)
        end = end.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT") if end.tzinfo is None else end.tz_convert(tz)

        df.loc[(df.index >= start) & (df.index <= end), shutdown["name"]] = 0.0
    return df

# --- Prognoza zużycia energii--- " 
def calculate_consumption_forecast() -> pd.DataFrame:
    """
    df: dataframe z kolumnami ['timestamp', 'Wolumen']
        timestamp powinien być tz-aware w strefie Europe/Warsaw i godzinowy
    forecast_days: liczba dni prognozy
    tz: strefa czasowa
    """

    # --- Wczytanie danych ---
    df = pd.read_csv(MEASUREMENTS_PATH, sep=";", decimal=",")

    # --- Konwersje ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(TZ)
    df['Wolumen'] = df['Wolumen'].astype(float)
    df['PPE'] = df['PPE'].astype(str)

    # --- Filtracja danych ---
    df = df[df['kierunek'] == 'CP']
    df = df[df['timestamp'].dt.minute == 0]
    df['Wolumen'] *= 4

    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour

    # --- Ostatnie obserwacje dla kombinacji dzień tygodnia + godzina + PPE ---
    df_sorted = df.sort_values('timestamp')
    last_values = (
        df_sorted.groupby(['weekday', 'hour', 'PPE'])['Wolumen']
        .last()
        .copy()
    )

    # --- Wartości z niedzieli (weekday = 6) ---
    last_sunday_hourly = last_values.xs(6, level='weekday')

    # --- Lista dni wolnych w Polsce ---
    years = df['timestamp'].dt.year.unique().tolist()
    pl_holidays = holidays.PL(years=years)

    # --- Generowanie dat prognozy godzinowej ---
    today = pd.Timestamp.now(tz=TZ).normalize()
    forecast_hours = pd.date_range(
        start=today,
        end=today + pd.Timedelta(days=DURATION_DAYS + 2),
        freq='H',
        tz=TZ
    )

    # --- Tworzenie prognozy dla każdego PPE ---
    forecast_list = []
    for ppe in df['PPE'].unique():
        forecast_df = pd.DataFrame({'timestamp': forecast_hours})
        forecast_df['PPE'] = ppe
        forecast_df['weekday'] = forecast_df['timestamp'].dt.weekday
        forecast_df['hour'] = forecast_df['timestamp'].dt.hour

        # Przypisanie wartości z historycznych danych
        forecast_df = forecast_df.merge(
            last_values.xs(ppe, level='PPE').reset_index(),
            on=['weekday', 'hour'],
            how='left'
        ).rename(columns={'Wolumen': 'forecast'})

        # --- Funkcja wyjątków ---
        def adjust_exceptions(row):
            date = row['timestamp'].date()
            year = row['timestamp'].year

            # Okres Bożego Narodzenia
            xmas_date = pd.Timestamp(f'{year}-12-25', tz=TZ)
            xmas_week_monday = xmas_date - pd.Timedelta(days=xmas_date.weekday() + 7)
            new_year_end = pd.Timestamp(f'{year+1}-01-02', tz=TZ)

            if xmas_week_monday <= row['timestamp'] <= new_year_end:
                if row['hour'] in last_sunday_hourly.xs(ppe, level='PPE').index:
                    return last_sunday_hourly.xs(ppe, level='PPE').loc[row['hour']]
                else:
                    return row['forecast']

            # Dni wolne
            if date in pl_holidays:
                if row['hour'] in last_sunday_hourly.xs(ppe, level='PPE').index:
                    return last_sunday_hourly.xs(ppe, level='PPE').loc[row['hour']]
                else:
                    return row['forecast']

            return row['forecast']

        forecast_df['forecast'] = forecast_df.apply(adjust_exceptions, axis=1)
        forecast_list.append(forecast_df[['timestamp', 'PPE', 'forecast']])

    # --- Łączenie wszystkich PPE ---
    forecast_all = pd.concat(forecast_list, ignore_index=True)

    # --- Zapis ---
    forecast_all.to_csv(MEASUREMENTS_FORECAST_PATH, index=False, sep=";", decimal=",", encoding="cp1250", float_format="%.6f")
    logging.info(f"Zapisano prognozę konsumpcji energii")
    
    return forecast_all

    

# --- Maksymalny potencjał generacji z PV --- " 
def generate_pv_potential(
    start="2024-06-14",
    end=(pd.Timestamp.today() + pd.DateOffset(months=50)).normalize() + pd.Timedelta(hours=23, minutes=45),
    freq="h",
    lat=50.332,
    lon=18.598,
    model="ineichen",
    shutdowns_file: Path = None
):
    
    times = pd.date_range(start, end, freq=freq, tz=TZ)
    location = pvlib.location.Location(lat, lon, tz=TZ)

    poa_df = pd.DataFrame(index=times)

    for inst in INSTALLATIONS:
        solar_position = location.get_solarposition(times)
        cs = location.get_clearsky(times, model=model)

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=inst["tilt"],
            surface_azimuth=inst["azimuth"],
            dni=cs["dni"],
            ghi=cs["ghi"],
            dhi=cs["dhi"],
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
        )

        poa_df[inst["name"]] = poa["poa_global"] * inst["capacity"]

    # zastosowanie masek czasowych, jeśli plik podany
    if shutdowns_file is not None and shutdowns_file.exists():
        masks = load_shutdowns(shutdowns_file)
        poa_df = apply_shutdowns(poa_df, masks)

    # --- agregacja per PPE ---
    df_ppe = pd.DataFrame(index=poa_df.index)
    for ppe in set(inst["ppe"] for inst in INSTALLATIONS):
        names = [inst["name"] for inst in INSTALLATIONS if inst["ppe"] == ppe]
        total = poa_df[names].sum(axis=1)
        cap = sum(inst["capacity"] for inst in INSTALLATIONS if inst["ppe"] == ppe)
        df_ppe[f"relative_{ppe}"] = total / total.max() * cap * 0.83

    df_ppe.index.name = "timestamp"
    df_ppe.to_csv(PV_MAX_POTENTIAL_CSV, index=True, sep=";", decimal=",", encoding="cp1250", float_format="%.6f")
    logging.info(f"Zapisano prognozę maksymalnego potencjału PV")
    return df_ppe

# --- PV prognoza z uwzględnieniem pogody ---
def generate_pv_generation_forecast(
    start="2024-06-14",
    end=(pd.Timestamp.today() + pd.DateOffset(months=2)).normalize() + pd.Timedelta(hours=23, minutes=45),
    freq="h",
    lat=50.332,
    lon=18.598,
    model="ineichen",
    shutdowns_file: Path = None,
    weather_file: Path = None,
) -> pd.DataFrame:
    """Generuje prognozę produkcji PV na podstawie prognozy pogody i konfiguracji instalacji."""
    times = pd.date_range(start, end, freq=freq, tz=TZ)
    location = pvlib.location.Location(lat, lon, tz=TZ)

    # Dane pogodowe
    weather = pd.read_csv(weather_file, sep=";", decimal=",", parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").reindex(times)

    # Brak prognozy → domyślne wartości
    weather["cloud_area_fraction"] = weather["cloud_area_fraction"].fillna(50)
    weather["temperature"] = weather["temperature"].fillna(15)
    numeric_cols = ["temperature", "cloud_area_fraction", "wind_speed"]

    # Wygładzenie braków
    weather[numeric_cols] = weather[numeric_cols].interpolate(method='linear', limit_direction='both')

    poa_df = pd.DataFrame(index=times)

    for inst in INSTALLATIONS:
        solar_position = location.get_solarposition(times)
        cs = location.get_clearsky(times, model=model)

        # Redukcja irradiancji o zachmurzenie (skalowanie 20–80)
        cloud_scaled = 20 + (weather["cloud_area_fraction"] / 100) * 60
        factor = 1 - cloud_scaled / 100.0

        ghi = cs["ghi"] * factor
        dni = cs["dni"] * factor
        dhi = cs["dhi"] * factor

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=inst["tilt"],
            surface_azimuth=inst["azimuth"],
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
        )

        # Temperatura modułu (model Faiman)
        temp_module = pvlib.temperature.faiman(
            poa_global=poa["poa_global"],
            temp_air=weather["temperature"],
            u0=25,
            u1=6,
        )

        # Moc skorygowana o temperaturę
        power = poa["poa_global"] * inst["capacity"]
        power_corr = power * (1 + GAMMA * (temp_module - 25))

        poa_df[inst["name"]] = power_corr

    # Zastosowanie masek czasowych
    if shutdowns_file and shutdowns_file.exists():
        masks = load_shutdowns(shutdowns_file)
        poa_df = apply_shutdowns(poa_df, masks)

    # Agregacja per PPE
    pv_forecast = pd.DataFrame(index=poa_df.index)
    for ppe in {inst["ppe"] for inst in INSTALLATIONS}:
        names = [inst["name"] for inst in INSTALLATIONS if inst["ppe"] == ppe]
        total = poa_df[names].sum(axis=1)
        cap = sum(inst["capacity"] for inst in INSTALLATIONS if inst["ppe"] == ppe)
        pv_forecast[f"forecast_{ppe}"] = total / total.max() * cap * 0.83

    pv_forecast.index.name = "timestamp"
    pv_forecast.to_csv(PV_FORECAST_CSV, index=True, sep=";", decimal=",", encoding="cp1250", float_format="%.6f")
    logging.info("Zapisano prognozę generacji energii z PV")
    
    return pv_forecast


def _export_sogl_csv(
    df: pd.DataFrame,
    col_name: str,
    code: str,
    date_to: datetime,
    consumption_df: pd.DataFrame = None 
) -> pd.DataFrame:
    """Eksportuje fragment prognozy do pliku w formacie SOGL z dynamiczną nazwą,
    uwzględniając prognozę zużycia (consumption_df) dla wyliczenia PAUTO.
    """


    # formatowanie dat do nazwy pliku
    today = datetime.now(TZ)
    today_str = today.strftime("%Y%m%d")
    date_to_str = date_to.strftime("%m%d")
    file_name = f"{today_str}_{date_to_str}_HVG_{code}.csv"

    # przygotowanie danych do eksportu
    df_out = df[["DATA I CZAS OD", col_name]].copy()
    df_out["DATA I CZAS OD"] = pd.to_datetime(df_out["DATA I CZAS OD"])
    df_out["PPLAN"] = df_out[col_name]


    # --- NOWY FRAGMENT: włączenie danych o zużyciu ---
    if consumption_df is not None:
        # dopasuj PPE po kodzie MWE (np. 33P0009 -> 590322400101395749)
        ppe_map = {
            "MWE_0940000_33P0009": "590322400101395749",
            "MWE_0940000_33P0010": "590322400100343246",
        }
        ppe = ppe_map.get(col_name)

        # upewnij się, że consumption_df jest w odpowiednim formacie
        consumption = consumption_df[consumption_df["PPE"] == ppe].copy()
        consumption = consumption.rename(columns={"timestamp": "DATA I CZAS OD", "forecast": "CONSUMPTION"})
        consumption['CONSUMPTION'] = consumption['CONSUMPTION']/1000
        consumption["DATA I CZAS OD"] = pd.to_datetime(consumption["DATA I CZAS OD"])

        # merge po czasie
        df_out = df_out.merge(consumption, on="DATA I CZAS OD", how="left")

        # wyliczenie PAUTO = max(PPLAN - CONSUMPTION, 0)
        df_out["PAUTO"] = np.maximum(df_out["PPLAN"] - df_out["CONSUMPTION"], 0)

    else:
        df_out["PAUTO"] = 0.000  # fallback, jeśli nie podano consumption_df

    # --- Formatowanie czasu i poprawka duplikatów 02:00 ---
    df_out["PAUTO"] = 0.000 #tymczasowo - i tak produkcja to 0
    df_out["PAUTO"] = df_out["PAUTO"].round(3)
    df_out["DATA I CZAS OD"] = df_out["DATA I CZAS OD"].dt.strftime("%d-%m-%Y %H:%M")
    mask_duplicates = df_out["DATA I CZAS OD"].duplicated(keep="first")
    df_out.loc[mask_duplicates, "DATA I CZAS OD"] = (
        df_out.loc[mask_duplicates, "DATA I CZAS OD"].str.replace("02:00", "02:00A")
    )

    df_out = df_out[["DATA I CZAS OD", "PPLAN", "PAUTO"]]

    # --- zapis do CSV ---
    path = SOGL_PATH / file_name
    with path.open("w", encoding="utf-8-sig") as f:
        f.write(f"#KOD_MWE;{col_name}\n")
    df_out.to_csv(path, index=False, sep=";", decimal=",", mode="a")
    logging.info(f"Zapisano prognozy do: {path}")

    return df_out



def prepare_forecasts_to_SOGL_CSV(pv_forecast: pd.DataFrame, consumption_df: pd.DataFrame):
    """Przygotowuje prognozy do formatu SOGL i eksportuje do CSV."""
    today = datetime.now(TZ)

    date_from = TZ.localize(datetime(today.year, today.month, today.day)) + timedelta(days=1)
    end_day = date_from + timedelta(days=DURATION_DAYS)
    date_to = TZ.localize(datetime(end_day.year, end_day.month, end_day.day, 23, 0, 0))

    df = pv_forecast[(pv_forecast.index >= date_from) & (pv_forecast.index <= date_to)].reset_index()

    df = df.rename(
        columns={
            "timestamp": "DATA I CZAS OD",
            "forecast_590322400101395749": "MWE_0940000_33P0009",
            "forecast_590322400100343246": "MWE_0940000_33P0010",
        }
    )

    for col in ["MWE_0940000_33P0009", "MWE_0940000_33P0010"]:
        df[col] = (df[col] / 1000).round(3)

    # przekazanie prognozy zużycia do eksportu
    df1 = _export_sogl_csv(df, "MWE_0940000_33P0009", "33P0009", date_to, consumption_df)
    df2 = _export_sogl_csv(df, "MWE_0940000_33P0010", "33P0010", date_to, consumption_df)

    
    return df1, df2

def _export_sogl_xml_full(
    df: pd.DataFrame,
    col_name: str,
    code: str,
    date_to: datetime,
    consumption_df: pd.DataFrame = None
) -> str:
    """Generuje XML PlannedResourceSchedule zgodny z SOGL, pozycje resetowane codziennie wg Europe/Warsaw."""

    # --- Przygotowanie danych ---
    df_out = df[["DATA I CZAS OD", col_name]].copy()
    df_out["DATA I CZAS OD"] = pd.to_datetime(df_out["DATA I CZAS OD"])
    df_out["PPLAN"] = df_out[col_name].clip(lower=0)

    # PAUTO z uwzględnieniem zużycia
    if consumption_df is not None:
        ppe_map = {
            "MWE_0940000_33P0009": "590322400101395749",
            "MWE_0940000_33P0010": "590322400100343246",
        }
        ppe = ppe_map.get(col_name)
        consumption = consumption_df[consumption_df["PPE"] == ppe].copy()
        consumption = consumption.rename(columns={"timestamp": "DATA I CZAS OD", "forecast": "CONSUMPTION"})
        consumption["CONSUMPTION"] = consumption["CONSUMPTION"] / 1000
        consumption["DATA I CZAS OD"] = pd.to_datetime(consumption["DATA I CZAS OD"])
        df_out = df_out.merge(consumption, on="DATA I CZAS OD", how="left")
        df_out["PAUTO"] = np.maximum(df_out["PPLAN"] - df_out["CONSUMPTION"].fillna(0), 0)
    else:
        df_out["PAUTO"] = 0.0

    df_out["PAUTO"] = df_out["PAUTO"].round(3)
    df_out["PPLAN"] = df_out["PPLAN"].round(3)

    # Konwersja do strefy Europe/Warsaw
    if df_out["DATA I CZAS OD"].dt.tz is None:
        df_out["DATA_LOCAL"] = df_out["DATA I CZAS OD"].dt.tz_localize("Europe/Warsaw")
    else:
        df_out["DATA_LOCAL"] = df_out["DATA I CZAS OD"].dt.tz_convert("Europe/Warsaw")

    # --- Nazwa pliku ---
    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    date_to_str = date_to.strftime("%m%d")
    file_name = f"{today_str}_{date_to_str}_HVG_{code}.xml"[:50]

    # --- Tworzenie XML ---
    root = ET.Element("PlannedResourceSchedule")
    ET.SubElement(root, "type").text = "A71"

    start_iso = df_out["DATA_LOCAL"].min().tz_convert("UTC").strftime("%Y-%m-%dT%H:00Z")
    # koniec okresu = ostatni punkt + 1h
    end_iso = (df_out["DATA_LOCAL"].max() + pd.Timedelta(hours=1)).tz_convert("UTC").strftime("%Y-%m-%dT%H:00Z")

    time_interval = ET.SubElement(root, "schedule_Period.timeInterval")
    ET.SubElement(time_interval, "start").text = start_iso
    ET.SubElement(time_interval, "end").text = end_iso

    # mRID jednostek MWE
    mwe_mrid_map = {
        "MWE_0940000_33P0009": "_ebc8ffff-1beb-4e6c-881c-b950600836fb",
        "MWE_0940000_33P0010": "_66b7dee0-eab7-4356-bef3-8102fc844e2e",
    }

    def add_series_daily(business_type: str, values: pd.Series, timestamps: pd.Series, series_mrid: str):
        ts = ET.SubElement(root, "PlannedResource_TimeSeries")
        ET.SubElement(ts, "mRID").text = series_mrid
        ET.SubElement(ts, "businessType").text = business_type
        ET.SubElement(ts, "measurement_Unit.name").text = "MAW"

        # przypisanie registeredResource.mRID
        ET.SubElement(ts, "registeredResource.mRID").text = (
            mwe_mrid_map.get(col_name, f"_{code}" if business_type == "A01" else "1")
        )

        df_tmp = pd.DataFrame({"timestamp": timestamps, "value": values})
        df_tmp["date_local"] = df_tmp["timestamp"].dt.tz_convert("Europe/Warsaw").dt.date

        for day, group in df_tmp.groupby("date_local"):
            # walidacja kompletności doby
            if len(group) not in (23, 24, 25):
                logging.warning(f"Doba {day} ma {len(group)} punktów – nietypowa liczba!")

            series_period = ET.SubElement(ts, "Series_Period")
            ti = ET.SubElement(series_period, "timeInterval")

            start_iso_day = group["timestamp"].min().tz_convert("UTC").strftime("%Y-%m-%dT%H:00Z")
            # <-- najważniejsza poprawka: koniec = ostatni punkt + 1h
            end_iso_day = (group["timestamp"].max() + pd.Timedelta(hours=1)).tz_convert("UTC").strftime("%Y-%m-%dT%H:00Z")

            ET.SubElement(ti, "start").text = start_iso_day
            ET.SubElement(ti, "end").text = end_iso_day
            ET.SubElement(series_period, "resolution").text = "PT1H"

            for idx, val in enumerate(group["value"], 1):
                val = max(val, 0)
                if business_type == "P01":
                    # PAUTO nie może przekraczać PPLAN
                    val = min(val, df_out.loc[df_out["DATA_LOCAL"] == group["timestamp"].iloc[idx - 1], "PPLAN"].values[0])
                point = ET.SubElement(series_period, "Point")
                ET.SubElement(point, "position").text = str(idx)
                ET.SubElement(point, "quantity").text = f"{val:.3f}"

    # --- Dodanie serii PPLAN i PAUTO ---
    add_series_daily("A01", df_out["PPLAN"], df_out["DATA_LOCAL"], "1")
    add_series_daily("P01", df_out["PAUTO"], df_out["DATA_LOCAL"], "2")

    # --- Formatowanie XML ---
    xml_str = ET.tostring(root, encoding="ISO-8859-2")
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="    ", encoding="ISO-8859-2")

    # usuń standardową deklarację i dodaj standalone="yes"
    pretty_xml_no_decl = b'\n'.join(pretty_xml_str.split(b'\n')[1:])
    xml_content = b'<?xml version="1.0" encoding="ISO-8859-2" standalone="yes"?>\n' + pretty_xml_no_decl

    # zapis pliku
    path = SOGL_PATH / file_name
    with path.open("wb") as f:
        f.write(xml_content)

    logging.info(f"Zapisano prognozy do: {path}")
    return xml_content




def prepare_forecasts_to_SOGL_XML(pv_forecast: pd.DataFrame, consumption_df: pd.DataFrame):
    """Przygotowuje prognozy do formatu SOGL i generuje pliki XML dla wszystkich MWE."""
    
    today = pd.Timestamp.now(TZ)
    date_from = TZ.localize(pd.Timestamp(today.year, today.month, today.day)) + pd.Timedelta(days=1)
    end_day = date_from + pd.Timedelta(days=DURATION_DAYS)
    date_to = TZ.localize(pd.Timestamp(end_day.year, end_day.month, end_day.day, 23, 0, 0))
    
    # filtr prognoz w zadanym okresie
    df = pv_forecast[(pv_forecast.index >= date_from) & (pv_forecast.index <= date_to)].reset_index()
    
    # dopasowanie nazw kolumn do MWE
    df = df.rename(
        columns={
            "timestamp": "DATA I CZAS OD",
            "forecast_590322400101395749": "MWE_0940000_33P0009",
            "forecast_590322400100343246": "MWE_0940000_33P0010",
        }
    )
    
    # konwersja do MW i zaokrąglenie
    for col in ["MWE_0940000_33P0009", "MWE_0940000_33P0010"]:
        df[col] = (df[col] / 1000).round(3)
    
    xml_files = {}
    for col, code in [("MWE_0940000_33P0009", "33P0009"), ("MWE_0940000_33P0010", "33P0010")]:
        xml_content = _export_sogl_xml_full(df, col, code, date_to, consumption_df)
        xml_files[code] = xml_content
    
    return xml_files


if __name__ == "__main__":
    
    logging.info("Start")
    
    # Pobierz dane pomiarowe dla Tauron
    run_data_download(['tauron_ewee', 'tauron_ewep'])
    
    # Przekształć dane pomiarowe do formatów użytecznych dla prognoz
    prepare_data.main()
    
    # Wyznacz prognozę zużycia energii
    consumption = calculate_consumption_forecast()
    
    # Przelicz maksymalną potencjalną generację z PV przy optymalnych warunkach pogodowych
    pv_potential = generate_pv_potential()
    
    # Wyznacz prognozę generacji PV
    pv_forecast = generate_pv_generation_forecast(shutdowns_file=SHUTDOWNS_CSV, weather_file=WEATHER_CSV)

    # Stwórz pliki do przeniesienia na portal Tauron
    df1_csv, df2_csv = prepare_forecasts_to_SOGL_CSV(pv_forecast, consumption) #csv
    df1_xml, df2_xml = prepare_forecasts_to_SOGL_XML(pv_forecast, consumption) #xml