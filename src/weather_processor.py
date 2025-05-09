from pathlib import Path
import os
import pandas as pd
from io import StringIO
from unidecode import unidecode
from scipy.interpolate import PchipInterpolator


# --------------------------------------------------
# Weather Processing
# --------------------------------------------------
class WeatherProcessor:
    @staticmethod
    def parse_metadata(folder: Path) -> dict:
        meta = {}
        for f in os.listdir(folder):
            if f.endswith(".txt"):
                lines = Path(folder / f).read_text().splitlines()[11:18]
                dfm = pd.read_csv(StringIO("\n".join(lines)))
                meta.update(dfm.set_index("Id_Sensore")["Nome_Sensore"].to_dict())
        return meta

    @staticmethod
    def read_raw(folder: Path, metadata: dict) -> pd.DataFrame:
        dfs = []
        for f in os.listdir(folder):
            if f.endswith(".csv"):
                dft = pd.read_csv(folder / f, index_col="Data-Ora", parse_dates=True)
                orig = [c for c in dft.columns if c != "Id Sensore"][0]
                sid = dft["Id Sensore"].iat[0]
                name = (
                    unidecode(f"{metadata[sid].strip()} {orig.strip()}")
                    .replace(" ", "_")
                    .lower()
                )
                dft.replace([777, 7777], 0, inplace=True)
                dft.replace([888, 8888, -999], pd.NA, inplace=True)
                dfs.append(
                    dft.rename(columns={orig: name}).drop(columns=["Id Sensore"])
                )
        return pd.concat(dfs, axis=1)

    @staticmethod
    def interpolate(df: pd.DataFrame) -> pd.DataFrame:
        t0 = df.index.view("int64") / 1e9
        t1 = pd.date_range(df.index[0], df.index[-1], freq="1S")
        t1s = t1.view("int64") / 1e9
        data = {
            col: PchipInterpolator(t0[df[col].notna()], df[col][df[col].notna()])(t1s)
            for col in df.columns
        }
        return pd.DataFrame(data, index=t1)
