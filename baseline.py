import pandas as pd
from pathlib import Path
import ruptures as rpt

'''
Used for dataloader2.py, the site to process parameters
'''
def gen_site_csv(site = None):
        if site==None:
                site_ids = [
                        "942030-99999",
                        "943350-99999",
                        "943740-99999",
                        "944760-99999",
                        "474250-99999",
                        "477590-99999",
                        "477710-99999",
                        "478030-99999",
                        "471100-99999",
                        "471080-99999",
                        "471420-99999",
                        "471510-99999",
                        "760500-99999",
                        "760610-99999",
                        "761130-99999",
                        "761220-99999",
                        "843900-99999",
                        "844520-99999",
                        "846910-99999",
                        "847520-99999",
                        "726810-24131",
                        "726815-24106",
                        "722860-23119",
                        "722265-13821",
                ]

                rows = []
                for sid in site_ids:
                    usaf, wban = sid.split("-")
                    rows.append({"USAF": usaf, "WBAN": wban})

                df = pd.DataFrame(rows)
                df.to_csv("sites_to_process.csv", index=False)

        else:
                print("check gen_site_csv() for supporting passing parameter")
                exit(1)

'''
path:  path to 23 sites data "weather_1901_2019_xxxx_xxxxx.csv"
return val: dict[site_id] = {"X":(T, d), "dates" = [list of timestamps}}. 
        dict[site_id]["X"] -> X(T, d), T is the time series size, d is num of featrues
'''
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [
        "mean_temp", "max_temp", "min_temp",
        "sea_level_pressure", "wind_speed", "precipitation"
    ]

    site_dict = {}

    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("date")

        X = g[feature_cols].to_numpy(dtype=float)
        dates = g["date"].to_list()

        site_dict[site_id] = {
            "X": X,          # shape (T, d)
            "dates": dates,  # list of timestamps
        }

    return site_dict

def run_pelt_one_site(X, model="rbf", penalty=10):
    """
    X: shape (T, d)
    model: 'rbf' (captures mean+variance changes) or 'l2' (mean changes)
    penalty: larger => fewer segments; smaller => more
    """
    algo = rpt.Pelt(model=model).fit(X)
    breakpoints = algo.predict(pen=penalty)   # e.g. [50, 120, 200, T]
    return breakpoints


def breakpoints_to_segments(dates, breakpoints):
        """
        dates: list of timestamps (length T)
        breakpoints: list of end indices from ruptures

        Returns list of dicts:
          [{'start_date': ..., 'end_date': ..., 'start_idx':..., 'end_idx':...}, ...]
        """
        segments = []
        prev = 0
        T = len(dates)

        for bp in breakpoints:
                end = min(bp, T)
                segments.append({
                        "start_idx": prev,
                        "end_idx": end - 1,  # inclusive
                        "start_date": dates[prev],
                        "end_date": dates[end - 1],
                })
                prev = end

        return segments

def run_pelt_all_sites(site_dict, model="rbf", penalty=10):
    all_rows = []

    for site_id, data in site_dict.items():
        X = data["X"]       # (T, d)
        dates = data["dates"]

        if len(X) < 2:
            # cannot segment
            continue

        # run PELT on this site
        # breakpoints
        bps = run_pelt_one_site(X, model=model, penalty=penalty)

        # turn breakpoints into time segments
        segments = breakpoints_to_segments(dates, bps)

        for seg_id, seg in enumerate(segments):
            all_rows.append({
                "site_id": site_id,
                "segment_id": seg_id,
                "start_idx": seg["start_idx"],
                "end_idx": seg["end_idx"],
                "start_date": seg["start_date"],
                "end_date": seg["end_date"],
                "length": seg["end_idx"] - seg["start_idx"] + 1,
            })

    return pd.DataFrame(all_rows)

if __name__ == "__main__":
        site_dict =  load_data("./weather_1901_2019_monthly_continuous.csv")
        segments = run_pelt_all_sites(site_dict)
        segments.to_csv("pelt_segments_enso24.csv", index=False)