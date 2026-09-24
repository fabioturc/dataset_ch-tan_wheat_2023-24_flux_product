"""
Project-wide settings for the CH-TAN wheat 2023-24 flux product.

All notebooks import the dataset period and data paths from here, so the period
is defined in one place only. The R scripts (50_REDDYPROC, 81.5) cannot import
this file; their dates are written out and must be kept consistent with it.

Timestamps
----------
All timestamps are CET (UTC+1, no daylight saving time). Records are 30 min.
The dataset period is defined by its boundaries:

    PERIOD_START = 2023-10-17 00:00  -> start of the first 30-min record
    PERIOD_END   = 2024-08-23 00:00  -> end of the last 30-min record (22 Aug 23:30-24:00)

i.e. it covers the full days 17 Oct 2023 (first management event) to 22 Aug 2024
(last tillage before sowing of the grass-legume mixture on 23 Aug 2024).
"""

from pathlib import Path
from typing import Union

import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # raw input data (downloaded from Zenodo)
NOTEBOOKS_DIR = ROOT / "notebooks"

# -----------------------------------------------------------------------------
# Site
# -----------------------------------------------------------------------------
SITE_ID = "CH-TAN"
SITE_LAT = 47.480620
SITE_LON = 8.911868
TIMEZONE = "Etc/GMT-1"            # CET, UTC+1 (note the inverted sign of Etc/ zones)

# -----------------------------------------------------------------------------
# Dataset period
# -----------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2023-10-17 00:00")
PERIOD_END = pd.Timestamp("2024-08-23 00:00")


def select_period(data: Union[pd.DataFrame, pd.Series],
                  timestamp: str = "middle",
                  start: pd.Timestamp = PERIOD_START,
                  end: pd.Timestamp = PERIOD_END,
                  verbose: bool = True) -> Union[pd.DataFrame, pd.Series]:
    """Restrict time series with a DatetimeIndex to the dataset period.

    Parameters
    ----------
    data : DataFrame or Series with a DatetimeIndex (naive CET or timezone-aware).
    timestamp : how the index labels each 30-min record:
        'middle' (e.g. 00:15) -> keeps start < t < end
        'end'    (e.g. 00:30) -> keeps start < t <= end
        'start'  (e.g. 00:00) -> keeps start <= t < end
    start, end : period boundaries in CET; default to the dataset period.
    """
    idx = data.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("select_period: data must have a DatetimeIndex")

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if idx.tz is not None:
        s = s.tz_localize(TIMEZONE).tz_convert(idx.tz)
        e = e.tz_localize(TIMEZONE).tz_convert(idx.tz)

    if timestamp == "middle":
        mask = (idx > s) & (idx < e)
    elif timestamp == "end":
        mask = (idx > s) & (idx <= e)
    elif timestamp == "start":
        mask = (idx >= s) & (idx < e)
    else:
        raise ValueError("select_period: timestamp must be 'middle', 'end' or 'start'")

    out = data.loc[mask].copy()
    if verbose:
        if len(out):
            print(f"select_period: kept {len(out)} of {len(data)} records "
                  f"({out.index.min()} to {out.index.max()}, timestamp={timestamp})")
        else:
            print(f"select_period: no records within {s} to {e}")
    return out
