import pandas as pd
import getpass
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
import re
import pytz
import logging
#import sys
import os

# --- Konfiguracja logowania ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

## --- Import z repozytorium ---
#sys.path.append(r"C:\Repos\electricity-consumption_tpa")
#from EE_measurements_downloader import run_data_download

# --- Ustawienia ---
user = getpass.getuser()

base_path = Path(f"C:/Users/{user}/EWE/SP_Dzial_RH - Zasoby")
measurements_path = base_path / "Dane pomiarowe/Energia/DaneOSD/Tauron"
measurements_output_file = base_path / "Projekty/SOGL prognozowanie/prognoza_Hirschvogel/dane_pomiarowe_Hirschvogel.csv"

weather_forecast_path = base_path / "Dane rynkowe/YR"
weather_forecast_output_file = base_path / "Projekty/SOGL prognozowanie/prognoza_Hirschvogel/prognozy_pogody.csv"

pv_generation_path = base_path / "Projekty/SOGL prognozowanie/dane_PV_Hirschvogel"
pv_generation_output_file = base_path / "Projekty/SOGL prognozowanie/prognoza_Hirschvogel/dane_PV_Hirschvogel.csv"

start_date = pd.to_datetime("20241001", format="%Y%m%d")
ppe_filter = ['590322400100343246', '590322400101395749'] #PPE klienta Hirschvogel

# Ustawiamy katalog, w którym znajduje się skrypt i credentials.txt
os.chdir(r"C:\Repos\electricity-consumption_tpa")

# --- Funkcje pomocnicze ---
def get_calendar(start_date, freq="15min"):
    """
    Generuje kalendarz z timestampami co freq od start_date do końca następnego miesiąca.
    """
    start = pd.Timestamp(start_date, tz="Europe/Warsaw")
    end_of_next_month = (pd.Timestamp.today(tz="Europe/Warsaw") + pd.offsets.MonthEnd(2)).normalize() + pd.Timedelta(hours=23, minutes=45)
    calendar = pd.date_range(start=start, end=end_of_next_month, freq=freq, tz="Europe/Warsaw")

    df_calendar = pd.DataFrame({"timestamp": calendar})
    df_calendar["data"] = df_calendar["timestamp"].dt.date
    df_calendar["nr_kwadransu"] = df_calendar.groupby("data").cumcount() + 1

    return df_calendar


def prepare_measurements_file(file_path: Path):
    """
    Przetwarza plik pomiarowy Tauron (CSV).
    """
    df = pd.read_csv(file_path, sep=";", header=0, skiprows=7, encoding="cp1250",
                     dtype={0: str}, low_memory=False)

    df.columns = ['PPE', 'kierunek', 'l_obs'] + list(df.columns[3:])
    date_str = file_path.stem.split("_")[-1]
    df['data'] = pd.to_datetime(date_str, format="%Y%m%d")

    id_vars = ['data', 'PPE', 'kierunek', 'l_obs']
    value_vars = df.columns[3:-1]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='nr_kwadransu', value_name='Wolumen')

    df_long['Wolumen'] = (
        df_long['Wolumen'].astype(str)
        .str.replace(r',[\+\-\*]$', '', regex=True)
        .astype(float)
    )
    df_long['l_obs'] = df_long['l_obs'].astype(int)
    df_long = df_long[df_long['PPE'].isin(ppe_filter)].reset_index(drop=True)

    mask = df_long['nr_kwadransu'].isin(['5A', '6A', '7A', '8A'])
    df_long.loc[mask, 'nr_kwadransu'] = df_long.loc[mask, 'nr_kwadransu'].str[0].astype(int) + 92
    df_long['nr_kwadransu'] = df_long['nr_kwadransu'].astype(int)

    conds = [
        (df_long["l_obs"] == 100) & (df_long["nr_kwadransu"].between(9, 96)),
        (df_long["l_obs"] == 100) & (df_long["nr_kwadransu"] > 96),
        (df_long["l_obs"] == 92) & (df_long["nr_kwadransu"] >= 13)
    ]
    choices = [
        df_long['nr_kwadransu'] + 4,
        df_long['nr_kwadransu'] - 88,
        df_long['nr_kwadransu'] - 4
    ]
    df_long['nr_kwadransu'] = np.select(conds, choices, default=df_long['nr_kwadransu'])
    df_long = df_long[df_long['l_obs'] >= df_long['nr_kwadransu']].drop(columns=['l_obs'])

    return df_long


def prepare_weather_forecast_file(file_path: Path):
    """
    Przetwarza plik prognozy pogody YR (CSV).
    """
    df = pd.read_csv(file_path, sep=";", decimal=',', header=0, encoding="cp1250", low_memory=False)
    df = df[['time', 'temperature', 'cloud_area_fraction', 'wind_speed', 'station']]
    df = df[df['station'].isin(['Biala', 'Katowice'])]

    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Warsaw')
    df = df.reset_index(drop=True)

    fname = file_path.stem  # np. yr_ewe_weather_forecast_2025-09-04_08-25-13
    parts = fname.split("_")
    sdatetime = parts[-2] + " " + parts[-1]
    df['file_date'] = pd.to_datetime(sdatetime, format="%Y-%m-%d %H-%M-%S")

    return df


def normalize_polish_date(s: str) -> str:
    """
    Zamienia polskie skróty miesięcy na numeryczne wartości.
    """
    miesiace = {
        "sty": "01", "lut": "02", "mar": "03", "kwi": "04", "maj": "05", "cze": "06",
        "lip": "07", "sie": "08", "wrz": "09", "paź": "10", "lis": "11", "gru": "12"
    }
    for m, num in miesiace.items():
        s = re.sub(rf"\b{m}\b", num, s, flags=re.IGNORECASE)
    return s


# --- Funkcje główne ---
def get_measurements():
    """
    Przetwarza dane pomiarowe Tauronu i generuje plik o nazwie dane_pomiarowe_Hirschvogel.csv
    """
    logging.info("Proces rozpoczęty")
    
    csv_files = [
        f for f in measurements_path.rglob("*.csv")
        if pd.to_datetime(f.stem.split("_")[-1], format="%Y%m%d") >= start_date
    ]
    logging.info(f"Znaleziono {len(csv_files)} plików pomiarowych")

    all_data = pd.concat([prepare_measurements_file(f) for f in tqdm(csv_files, desc="Przetwarzanie pomiarów")], ignore_index=True)
    all_data['data'] = pd.to_datetime(all_data['data']).dt.date

    df_calendar = get_calendar(start_date, "15min")
    all_data = all_data.merge(df_calendar, on=['data', 'nr_kwadransu'])
    all_data.sort_values(by=['PPE', 'kierunek', 'timestamp'], inplace=True)
    all_data = all_data[['timestamp', 'PPE', 'kierunek', 'Wolumen']]
    
    #all_data = all_data.set_index("timestamp").resample("1H").sum().reset_index()
    all_data.to_csv(measurements_output_file, index=False, sep=';', decimal=',', encoding='cp1250')

    logging.info(f"Przetwarzanie danych pomiarowych zakończone. Wynik zapisano w: {measurements_output_file}")


def get_weather_forecast_data():
    """
    Przetwarza dane pogodowe i generuje plik o nazwie prognozy_pogody.csv
    """
    csv_files_all = list(weather_forecast_path.rglob("*.csv"))
    csv_files = []
    for f in csv_files_all:
        try:
            date_str = f.stem.split("_")[-2]
            file_date = pd.to_datetime(date_str, format="%Y-%m-%d")
            if file_date >= start_date - timedelta(days=5):
                csv_files.append(f)
        except Exception:
            continue

    logging.info(f"Znaleziono {len(csv_files)} plików pogodowych")

    all_data = pd.concat([prepare_weather_forecast_file(f) for f in tqdm(csv_files, desc="Przetwarzanie prognoz pogody")], ignore_index=True)
    all_data.rename(columns={'time': 'timestamp'}, inplace=True)
    all_data.sort_values(by=['timestamp', 'station', 'file_date'], ascending=[True, False, False], inplace=True)
    all_data.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    all_data = all_data[all_data['timestamp'] >= pd.to_datetime(start_date).tz_localize('Europe/Warsaw')]
    all_data.drop(columns=['file_date'], inplace=True)
    all_data = all_data.set_index('timestamp')
    all_data[['temperature', 'cloud_area_fraction', 'wind_speed']] = all_data[['temperature', 'cloud_area_fraction', 'wind_speed']].astype(float)
    all_data = all_data.resample('h').asfreq()
    all_data.interpolate(method='linear', inplace=True)
    all_data = all_data.round(3).ffill()
    
    #all_data['data'] = all_data.index.date
    #all_data["nr_kwadransu"] = all_data.groupby("data").cumcount() + 1

    all_data.to_csv(weather_forecast_output_file, index=True, float_format="%.6f", sep=';', decimal=',', encoding='cp1250')
    logging.info(f"Prognozy pogody zapisano w: {weather_forecast_output_file}")


def get_pv_generation():
    """
    Przetwarza dane o generacji PV i generuje plik o nazwie dane_PV_Hirschvogel.csv
    """
    tz = "Europe/Warsaw"
    pattern = r"energy-chart-data (\d{2}_\d{2}_\d{4} \d{2}_\d{2} (?:AM|PM))"
    csv_files = list(pv_generation_path.glob("*.csv"))

    dfs = []
    for file in tqdm(csv_files, desc="Przetwarzanie plików PV"):
        match = re.search(pattern, file.name)
        if match:
            dt_str = match.group(1)
            czas_pobrania = datetime.strptime(dt_str, "%m_%d_%Y %I_%M %p")
            czas_pobrania = pytz.timezone(tz).localize(czas_pobrania)
        else:
            czas_pobrania = None

        df = pd.read_csv(file, sep=",")
        if "Produkcja (Wh)" in df.columns:
            df["Produkcja (kWh)"] = df["Produkcja (Wh)"] / 1000.0
            df.drop(columns=["Produkcja (Wh)"], inplace=True)

        # szybkie parsowanie dat PL
        df["Czas pomiaru"] = df["Czas pomiaru"].map(normalize_polish_date)
        df["Czas pomiaru"] = pd.to_datetime(df["Czas pomiaru"], format="%d %m %Y %H:%M", errors="coerce")
        df.dropna(subset=["Czas pomiaru"], inplace=True)
        df["Czas pomiaru"] = df["Czas pomiaru"].dt.tz_localize(tz, ambiguous=True)

        if czas_pobrania:
            df = df[df["Czas pomiaru"] < czas_pobrania]

        dfs.append(df[["Czas pomiaru", "Produkcja (kWh)"]])

    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.sort_values("Produkcja (kWh)").drop_duplicates("Czas pomiaru", keep="last")
    all_data = all_data.sort_values("Czas pomiaru").reset_index(drop=True)
    all_data = all_data.rename(columns={"Czas pomiaru":"timestamp"})
    all_data = all_data.set_index("timestamp").resample("h").sum().reset_index()
    all_data.to_csv(pv_generation_output_file, index=False, float_format="%.6f", sep=";", decimal=",", encoding="cp1250")
    logging.info(f"Dane PV zapisano w: {pv_generation_output_file}")

def main():

    get_measurements()
    get_weather_forecast_data()
    #Niezaimplementowane pobieranie danych o generacji PV przez API
    get_pv_generation()

if __name__ == "__main__":
    main()