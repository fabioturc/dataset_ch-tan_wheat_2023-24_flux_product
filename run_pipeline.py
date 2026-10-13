#!/usr/bin/env python
"""
Run the processing steps of this flux product in order, without opening the notebooks.

Run from the repository root, with the Python environment of environment.yml active:

    python run_pipeline.py                  # all steps (13 -> 90, see step 12 below)
    python run_pipeline.py --list           # show the steps and exit
    python run_pipeline.py --from 41        # start at step 41 (41, 42, ..., 90)
    python run_pipeline.py --from 81 --to 84
    python run_pipeline.py --only 71 90     # only these steps
    python run_pipeline.py --skip-tuning    # skip 8x.2 and 8x.3 (feature selection, hyperparameters)
    python run_pipeline.py --dry-run        # show what would run
    python run_pipeline.py --status         # show when the output of each step was written

A step is given by its number: "41" is notebook 41.0, "81.4" is notebook 81.4.0, and "81"
stands for all steps 81.x. Each notebook runs in a fresh kernel with its own folder as
working directory (as when opened in Jupyter or VS Code) and is saved in place with its
outputs, so it can be opened afterwards to check plots and prints. The run stops at the
first error; the failing notebook is saved up to the failing cell.

The two R steps (51 and 81.5) are run with Rscript. Rscript is searched on the PATH and, on
Windows, in "Program Files/R", in "AppData/Local/Programs/R" (per-user installation) and in
the registry; otherwise pass it with --rscript or the RSCRIPT environment variable.
The R Markdown file of step 51 is rendered to HTML if pandoc is available (e.g. from an
RStudio installation), otherwise it is knitted to Markdown; the results are the same.

Step 12 writes the gap-filled meteo file in data/METEO, which is part of the raw data of the
published dataset. It therefore runs only when selected explicitly (--from 12 or --only 12) or
when that file does not exist yet; it also needs the internal file 11.1_..._diive.csv of step 11.

Everything printed is also written to run_pipeline.log in the repository root.
"""

import argparse
import asyncio
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- Product ---------------------------------------------------------------------------
PRODUCT = "CH-TAN wheat 2023-24 flux product"
TAG = "wheat_2023-24"        # in the names of the final product files
METEO_TAG = "2023-24"         # in the name of the gap-filled meteo file (output of step 12)
EXTRA_FINAL_OUTPUTS = [  # full product as csv, subset for the analysis repository n2o-precision-fertilization
    "notebooks/90_FINAL_MERGE/90.1_CH-TAN_wheat_2023-24_FLUXES_METEO_MGMT_FULL.csv",
    "notebooks/90_FINAL_MERGE/90.3_CH-TAN_wheat_2023-24_FLUXES_METEO_MGMT_SUBSET_n2o-precision-fertilization.parquet"]

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
LOGFILE = ROOT / "run_pipeline.log"

# (step, path relative to the repository root); listed in run order.
# Step 11 (download from the internal Swiss FluxNet database) is not part of the run:
# its output 11.1_CH-TAN_meteo_meteoscreening_diive.csv is an input of step 12.
STEPS = [
    ("12", "notebooks/10_METEO/12.0_GapFillingMeteoswiss.ipynb"),
    ("13", "notebooks/10_METEO/13.0_GapFillingMeteoXGBoost.ipynb"),
    ("21", "notebooks/20_MANAGEMENT/21.0_ConvertMgmtToTimeseries.ipynb"),
    ("31", "notebooks/30_MERGE_DATA/31.0_FLUXES_L1_IRGA.ipynb"),
    ("32", "notebooks/30_MERGE_DATA/32.0_FLUXES_L1_LGR.ipynb"),
    ("33", "notebooks/30_MERGE_DATA/33.0_FLUXES_L1_IRGA+LGR.ipynb"),
    ("41", "notebooks/40_FLUX_PROCESSING_CHAIN/41_NEE/41.0_FluxProcessingChain_L3.2_NEE.ipynb"),
    ("42", "notebooks/40_FLUX_PROCESSING_CHAIN/42_LE/42.0_FluxProcessingChain_L3.2_LE.ipynb"),
    ("43", "notebooks/40_FLUX_PROCESSING_CHAIN/43_H/43.0_FluxProcessingChain_L3.2_H.ipynb"),
    ("44", "notebooks/40_FLUX_PROCESSING_CHAIN/44_FN2O/44.0_FluxProcessingChain_L3.2_FN2O.ipynb"),
    ("45", "notebooks/40_FLUX_PROCESSING_CHAIN/45_FCH4/45.0_FluxProcessingChain_L3.2_FCH4.ipynb"),
    ("46", "notebooks/40_FLUX_PROCESSING_CHAIN/46.0_MergeFluxProcessingChainResults.ipynb"),
    ("51", "notebooks/50_REDDYPROC/51.0_UstarDetection_MDS_Gapfilling_NEE_Partitioning.Rmd"),
    ("61", "notebooks/60_USTAR_FILTERING/61.0_MergeFluxProcessingChainResults+REddyProcResults.ipynb"),
    ("62", "notebooks/60_USTAR_FILTERING/62.0_UstarFiltering_L3.3.ipynb"),
    ("71", "notebooks/70_SPLIT_TREATMENTS/71.0_AttributionTreatments.ipynb"),
    ("81.1", "notebooks/80_GAP-FILLING/81_NEE/81.1.0_PrepareInputData.ipynb"),
    ("81.2", "notebooks/80_GAP-FILLING/81_NEE/81.2.0_FeatureSelection.ipynb"),
    ("81.3", "notebooks/80_GAP-FILLING/81_NEE/81.3.0_HyperparameterOptimization.ipynb"),
    ("81.4", "notebooks/80_GAP-FILLING/81_NEE/81.4.0_GapFilling.ipynb"),
    ("81.5", "notebooks/80_GAP-FILLING/81_NEE/81.5.0_PartitioningREddyProc.R"),
    ("81.6", "notebooks/80_GAP-FILLING/81_NEE/81.6.0_CollectGapFillingPartitioningResults.ipynb"),
    ("82.1", "notebooks/80_GAP-FILLING/82_FN2O/82.1.0_PrepareInputData.ipynb"),
    ("82.2", "notebooks/80_GAP-FILLING/82_FN2O/82.2.0_FeatureSelection.ipynb"),
    ("82.3", "notebooks/80_GAP-FILLING/82_FN2O/82.3.0_HyperparameterOptimization.ipynb"),
    ("82.4", "notebooks/80_GAP-FILLING/82_FN2O/82.4.0_GapFilling.ipynb"),
    ("83.1", "notebooks/80_GAP-FILLING/83_FCH4/83.1.0_PrepareInputData.ipynb"),
    ("83.2", "notebooks/80_GAP-FILLING/83_FCH4/83.2.0_FeatureSelection.ipynb"),
    ("83.3", "notebooks/80_GAP-FILLING/83_FCH4/83.3.0_HyperparameterOptimization.ipynb"),
    ("83.4", "notebooks/80_GAP-FILLING/83_FCH4/83.4.0_GapFilling.ipynb"),
    ("84", "notebooks/80_GAP-FILLING/84.0_MergeGapFillingResults.ipynb"),
    ("90", "notebooks/90_FINAL_MERGE/90.0_MergeFluxesMeteoManagement.ipynb"),
]

# Main output file(s) of each step, relative to the repository root (used by --status)
_GF = "notebooks/80_GAP-FILLING"
_FINAL = "notebooks/90_FINAL_MERGE/90.{n}_CH-TAN_" + TAG + "_FLUXES_METEO_MGMT_{kind}"
OUTPUTS = {
    "12": [f"data/METEO/CH-TAN_meteo_gapfilled-meteoswiss_{METEO_TAG}.csv"],
    "13": ["notebooks/10_METEO/13.1_CH-TAN_meteo_gapfilled.parquet"],
    "21": ["notebooks/20_MANAGEMENT/21.1_mgmt_full_timestamp.parquet"],
    "31": ["notebooks/30_MERGE_DATA/31.1_CH-TAN_IRGA_Level-1_eddypro_fluxnet.parquet"],
    "32": ["notebooks/30_MERGE_DATA/32.1_CH-TAN_LGR_Level-1_eddypro_fluxnet.parquet"],
    "33": ["notebooks/30_MERGE_DATA/33.1_CH-TAN_IRGA+LGR_Level-1_eddypro_fluxnet.parquet"],
    "41": ["notebooks/40_FLUX_PROCESSING_CHAIN/41_NEE/41.1_FluxProcessingChain_L3.2_NEE.parquet"],
    "42": ["notebooks/40_FLUX_PROCESSING_CHAIN/42_LE/42.1_FluxProcessingChain_L3.2_LE.parquet"],
    "43": ["notebooks/40_FLUX_PROCESSING_CHAIN/43_H/43.1_FluxProcessingChain_L3.2_H.parquet"],
    "44": ["notebooks/40_FLUX_PROCESSING_CHAIN/44_FN2O/44.1_FluxProcessingChain_L3.2_FN2O.parquet"],
    "45": ["notebooks/40_FLUX_PROCESSING_CHAIN/45_FCH4/45.1_FluxProcessingChain_L3.2_FCH4.parquet"],
    "46": ["notebooks/40_FLUX_PROCESSING_CHAIN/46.1_FLUXES_L3.2_NEE_LE_H_FN2O_FCH4.parquet",
           "notebooks/40_FLUX_PROCESSING_CHAIN/46.2_FluxProcessingChain_L3.2_subset-forREddyProc.csv"],
    "51": ["notebooks/50_REDDYPROC/51.1_UstarThresholds_MDS-gapfilled_NEE-partitioned.csv"],
    "61": ["notebooks/60_USTAR_FILTERING/61.1_FLUXES_L3.2_NEE_LE_H_FN2O_FCH4_REDDYPROC.parquet"],
    "62": ["notebooks/60_USTAR_FILTERING/62.1_FLUXES_L3.3_NEE_LE_H_FN2O_FCH4_REDDYPROC.parquet"],
    "71": ["notebooks/70_SPLIT_TREATMENTS/71.1_FLUXES_L3.3_NEE_LE_H_FN2O_FCH4_REDDYPROC_PARCELS.parquet"],
    "84": [f"{_GF}/84.1_GF-FLUXES+PARTITIONED.parquet"],
    "90": [_FINAL.format(n=1, kind="FULL.parquet"), _FINAL.format(n=2, kind="CORE.parquet"),
           _FINAL.format(n=2, kind="CORE.csv")] + EXTRA_FINAL_OUTPUTS,
}
for _n, _gas, _folder in [("81", "NEE", "81_NEE"), ("82", "FN2O", "82_FN2O"), ("83", "FCH4", "83_FCH4")]:
    OUTPUTS[f"{_n}.1"] = [f"{_GF}/{_folder}/{_n}.1.1_GapFillingDataset.parquet"]
    OUTPUTS[f"{_n}.2"] = [f"{_GF}/{_folder}/best_features_{_gas}_XGBoost.txt"]
    OUTPUTS[f"{_n}.3"] = [f"{_GF}/{_folder}/best_hyperparameters_{_gas}_XGBoost.json"]
    OUTPUTS[f"{_n}.4"] = [f"{_GF}/{_folder}/{_n}.4.1_{_gas}_GF-XGBoost.parquet"]
OUTPUTS["81.4"].append(f"{_GF}/81_NEE/81.4.2_PartitioningSubsetForREddyProc.csv")
OUTPUTS["81.5"] = [f"{_GF}/81_NEE/81.5.1_NEE_XG-GAPF_PART_ReddyProc.csv"]
OUTPUTS["81.6"] = [f"{_GF}/81_NEE/81.6.1_NEE_GF-XGBoost_GPP_RECO.parquet"]

# Feature selection and hyperparameter optimisation: slow, and their results
# (best_features_*.txt, best_hyperparameters_*.json) are read by 8x.4.
TUNING_STEPS = {"81.2", "81.3", "82.2", "82.3", "83.2", "83.3"}
TUNING_OUTPUTS = {
    "81": ["best_features_NEE_XGBoost.txt", "best_hyperparameters_NEE_XGBoost.json"],
    "82": ["best_features_FN2O_XGBoost.txt", "best_hyperparameters_FN2O_XGBoost.json"],
    "83": ["best_features_FCH4_XGBoost.txt", "best_hyperparameters_FCH4_XGBoost.json"],
}

# Step 12 needs the screened station meteo of step 11 (internal database) and writes the
# gap-filled meteo file in data/METEO (raw data of the published dataset).
INPUT_12 = NB / "10_METEO" / "11.1_CH-TAN_meteo_meteoscreening_diive.csv"
OUTPUT_12 = ROOT / OUTPUTS["12"][0]

log = logging.getLogger("run_pipeline")


# -----------------------------------------------------------------------------
# Step selection
# -----------------------------------------------------------------------------
def _matches(step_id, arg):
    """'81' matches 81.1 ... 81.6; '81.4' (or '81.4.0') only 81.4; '41' (or '41.0') only 41."""
    arg = arg.strip()
    if arg.endswith(".0"):
        arg = arg[:-2]
    return step_id == arg or step_id.startswith(arg + ".")


def _positions(arg):
    pos = [i for i, (sid, _) in enumerate(STEPS) if _matches(sid, arg)]
    if not pos:
        raise SystemExit(f"Unknown step '{arg}'. Use --list to see the steps.")
    return pos


def select_steps(args):
    if args.only:
        keep = sorted({i for a in args.only for i in _positions(a)})
        steps = [STEPS[i] for i in keep]
    else:
        first = min(_positions(args.start)) if args.start else 0
        last = max(_positions(args.end)) if args.end else len(STEPS) - 1
        if first > last:
            raise SystemExit("--from must come before --to")
        steps = STEPS[first:last + 1]
    if args.skip_tuning:
        steps = [s for s in steps if s[0] not in TUNING_STEPS]
    return steps


def show_status():
    """Print, for each step, when its output was last written (or which output is missing)."""
    print(f"{PRODUCT}: output of each step, last modified")
    for sid, path in STEPS:
        files = [ROOT / f for f in OUTPUTS.get(sid, [])]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            state = "MISSING: " + ", ".join(missing)
        else:
            state = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(f.stat().st_mtime for f in files)))
        tag = "  [tuning]" if sid in TUNING_STEPS else ""
        print(f"  {sid:<5} {Path(path).name + tag:<70} {state}")


def check_tuning_outputs(steps):
    """With --skip-tuning, 8x.4 needs the files written by 8x.2 and 8x.3."""
    missing = []
    for sid, path in steps:
        if sid.endswith(".4") and sid[:2] in TUNING_OUTPUTS:
            folder = (ROOT / path).parent
            missing += [str((folder / f).relative_to(ROOT)) for f in TUNING_OUTPUTS[sid[:2]]
                        if not (folder / f).is_file()]
    if missing:
        raise SystemExit("--skip-tuning: these files are missing, run steps 8x.2 and 8x.3 first:\n  "
                         + "\n  ".join(missing))


# -----------------------------------------------------------------------------
# Environment check
# -----------------------------------------------------------------------------
class EnvironmentProblem(Exception):
    pass


def check_environment(kernel):
    """Fail early if this Python or the notebook kernel lacks the packages of the pipeline.
    Returns the Python executable used by the kernel."""
    import importlib.util
    missing = [m for m in ("nbformat", "nbclient", "ipykernel") if importlib.util.find_spec(m) is None]
    if missing:
        raise EnvironmentProblem(
            f"This Python ({sys.executable}) does not have: {', '.join(missing)}.\n"
            "Run the pipeline from the project's conda environment, e.g.\n"
            "  conda activate <environment>\n"
            "  python run_pipeline.py\n"
            "(environment.yml creates it as 'ch-tan-flux').")

    import nbformat
    from nbclient import NotebookClient
    kernel_name = kernel or "python3"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        "import sys\nprint(sys.executable)\nimport diive, xgboost, shap, windrose, pyarrow, seaborn"))
    try:
        NotebookClient(nb, kernel_name=kernel_name, timeout=300).execute()
    except Exception as err:  # noqa: BLE001
        raise EnvironmentProblem(
            f"The Jupyter kernel '{kernel_name}' cannot import the packages of the notebooks:\n{err}\n"
            "Activate the project environment before running, or choose its kernel with --kernel NAME "
            "(list kernels with: jupyter kernelspec list).") from None
    out = [o for o in nb.cells[0].outputs if o.get("output_type") == "stream"]
    return out[0]["text"].strip() if out else "unknown"


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
SAVE_EVERY_S = 60  # while a notebook runs, save its outputs at most this often


def run_notebook(path, step, kernel=None):
    import threading
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    n_cells = len(nb.cells)
    t_start = time.time()
    state = {"cell": 0, "saved": time.time()}
    stop = threading.Event()

    def ticker():
        # console only (not in the log file): current cell and elapsed time, refreshed every 2 s
        while True:
            print(f"\r    step {step}: cell {state['cell']}/{n_cells}, "
                  f"{fmt_duration(time.time() - t_start)} ", end="", flush=True)
            if stop.wait(2):
                break

    def on_start(cell, cell_index, **kwargs):
        state["cell"] = cell_index + 1

    def on_executed(cell, cell_index, **kwargs):
        # save regularly, so that the outputs so far (plots, prints) can be looked at during the run
        if time.time() - state["saved"] > SAVE_EVERY_S:
            nbformat.write(nb, path)
            state["saved"] = time.time()

    client = NotebookClient(
        nb,
        timeout=None,                 # no time limit per cell (gap-filling cells take long)
        kernel_name=kernel or "",     # "" = kernel named in the notebook (python3)
        allow_errors=False,
        record_timing=True,
        resources={"metadata": {"path": str(path.parent)}},  # working directory = notebook folder
        on_cell_start=on_start,
        on_cell_executed=on_executed,
    )
    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        client.execute()
    except CellExecutionError as err:
        # nbclient's message holds the failing cell's source and the traceback
        raise RuntimeError(f"error in cell {state['cell']} of {path.name}:\n{err}") from None
    finally:
        stop.set()
        thread.join()
        print("\r" + " " * 60 + "\r", end="", flush=True)
        nbformat.write(nb, path)      # saved with outputs, also when a cell failed


def _r_version_key(path):
    """'.../R-4.10.1/bin/Rscript.exe' -> (4, 10, 1), to pick the newest installation."""
    name = Path(path).parent.parent.name  # R-x.y.z
    try:
        return tuple(int(p) for p in name.split("-", 1)[1].split("."))
    except (IndexError, ValueError):
        return (0,)


def _rscript_from_registry():
    """R installer records its folder under Software\\R-core\\R (per user or for all users)."""
    try:
        import winreg
    except ImportError:
        return []
    found = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for key in (r"Software\R-core\R64", r"Software\R-core\R"):
            try:
                with winreg.OpenKey(hive, key) as k:
                    install_path = winreg.QueryValueEx(k, "InstallPath")[0]
                found.append(str(Path(install_path) / "bin" / "Rscript.exe"))
            except OSError:
                continue
    return found


def find_rscript(cli_value):
    if cli_value:
        if not Path(cli_value).is_file():
            raise SystemExit(f"--rscript: file not found: {cli_value}")
        return str(cli_value)
    for candidate in (os.environ.get("RSCRIPT"), shutil.which("Rscript")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    if sys.platform.startswith("win"):
        # all-users installs (Program Files) and per-user installs (AppData\Local\Programs)
        roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
        roots.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"))
        found = []
        for root in filter(None, roots):
            found += glob.glob(os.path.join(root, "R", "R-*", "bin", "Rscript.exe"))
        if found:
            return max(found, key=_r_version_key)  # newest installed version
        found = [p for p in _rscript_from_registry() if Path(p).is_file()]
        if found:
            return found[0]
    return None


def find_pandoc_dir():
    """pandoc shipped with RStudio, for rmarkdown::render outside RStudio."""
    if os.environ.get("RSTUDIO_PANDOC") or shutil.which("pandoc"):
        return None
    patterns = [
        r"C:\Program Files\RStudio\resources\app\bin\quarto\bin\tools\pandoc.exe",
        r"C:\Program Files\RStudio\resources\app\bin\quarto\bin\tools\*\pandoc.exe",
        r"C:\Program Files\RStudio\bin\quarto\bin\tools\pandoc.exe",
        r"C:\Program Files\RStudio\bin\pandoc\pandoc.exe",
        "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/*/pandoc",
        "/usr/lib/rstudio/resources/app/bin/quarto/bin/tools/pandoc",
    ]
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            return str(Path(found[0]).parent)
    return None


def run_command(cmd, cwd, env=None):
    log.info("  $ %s   (in %s)", " ".join(f'"{c}"' if " " in c else c for c in cmd), cwd)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log.info("  | %s", line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def run_rmarkdown(path, rscript):
    env = os.environ.copy()
    pandoc_dir = find_pandoc_dir()
    if pandoc_dir:
        env["RSTUDIO_PANDOC"] = pandoc_dir
    expr = (
        f"f <- '{path.name}'; "
        "if (requireNamespace('rmarkdown', quietly = TRUE) && rmarkdown::pandoc_available()) {"
        " rmarkdown::render(f, output_format = 'html_document', envir = new.env())"
        "} else {"
        " message('pandoc not found: knitting to Markdown instead of HTML');"
        " knitr::knit(f, envir = new.env())"
        "}"
    )
    run_command([rscript, "-e", expr], cwd=path.parent, env=env)


def run_rscript(path, rscript):
    # 81.5.0 uses paths relative to the repository root
    run_command([rscript, str(path.relative_to(ROOT)).replace("\\", "/")], cwd=ROOT)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGFILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def fmt_duration(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for details.")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--from", dest="start", metavar="STEP", help="first step to run, e.g. 41 or 81.4")
    parser.add_argument("--to", dest="end", metavar="STEP", help="last step to run, e.g. 62")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps, e.g. 71 90")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="skip feature selection and hyperparameter optimisation (8x.2, 8x.3)")
    parser.add_argument("--dry-run", action="store_true", help="show the steps that would run and exit")
    parser.add_argument("--status", action="store_true",
                        help="show when the output of each step was last written and exit")
    parser.add_argument("--kernel", help="Jupyter kernel name (default: the one in the notebooks, python3)")
    parser.add_argument("--rscript", help="path to Rscript.exe (default: searched on PATH, RSCRIPT variable, "
                                          "Program Files, AppData\\Local\\Programs, registry)")
    args = parser.parse_args()

    if args.list:
        print("Steps (run in this order):")
        for sid, path in STEPS:
            tag = "  [tuning]" if sid in TUNING_STEPS else ""
            print(f"  {sid:<5} {path}{tag}")
        print("\nStep 11 (internal database download) is not run; its output is an input of step 12.")
        return 0
    if args.status:
        show_status()
        return 0

    steps = select_steps(args)
    note_12 = None
    if steps and steps[0][0] == "12":
        asked_12 = bool(args.only and any(_matches("12", a) for a in args.only)) or \
                   bool(args.start and _matches("12", args.start))
        if OUTPUT_12.is_file() and not asked_12:
            steps = steps[1:]
            note_12 = (f"Step 12 skipped: {OUTPUT_12.relative_to(ROOT)} exists (raw data of the dataset). "
                       "To recreate it, run with --from 12 or --only 12.")
        elif not INPUT_12.is_file():
            if not OUTPUT_12.is_file():
                raise SystemExit(f"Step 12 needs {INPUT_12.relative_to(ROOT)} (internal) and "
                                 f"{OUTPUT_12.relative_to(ROOT)} (raw data) is missing too.")
            steps = steps[1:]
            note_12 = (f"Step 12 skipped: {INPUT_12.name} not available, "
                       f"using {OUTPUT_12.relative_to(ROOT)} from the raw data.")
    if not steps:
        raise SystemExit("No steps selected.")
    if args.skip_tuning:
        check_tuning_outputs(steps)

    rscript = None
    if any(p.endswith((".R", ".Rmd")) for _, p in steps):
        rscript = find_rscript(args.rscript)
        if rscript is None and not args.dry_run:
            raise SystemExit("Rscript not found (needed for steps 51 and 81.5). Pass it with --rscript, e.g.\n"
                             "  --rscript \"C:/Users/<you>/AppData/Local/Programs/R/R-4.2.2/bin/Rscript.exe\"\n"
                             "or run the steps before R only, e.g. --to 46.")

    if args.dry_run:
        if note_12:
            print(note_12)
        print("Would run:")
        for sid, path in steps:
            print(f"  {sid:<5} {path}")
        if any(p.endswith((".R", ".Rmd")) for _, p in steps):
            print(f"\nRscript: {rscript or 'NOT FOUND'}")
        return 0

    if not (ROOT / "data").is_dir():
        raise SystemExit("data/ folder not found in the repository root (see README, 'Reproducing the dataset').")

    if sys.platform.startswith("win"):
        # pyzmq (Jupyter kernels) does not work with the default Proactor event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    setup_logging()
    log.info("=" * 78)
    log.info("%s: running %d step(s): %s", PRODUCT, len(steps), ", ".join(s for s, _ in steps))
    log.info("Python: %s", sys.executable)
    if note_12:
        log.info(note_12)
    if rscript:
        log.info("Rscript: %s", rscript)

    if any(p.endswith(".ipynb") for _, p in steps):
        log.info("Checking the Python environment and the Jupyter kernel ...")
        try:
            kernel_python = check_environment(args.kernel)
        except EnvironmentProblem as err:
            log.info("STOPPED: %s", err)
            return 1
        log.info("Kernel Python: %s", kernel_python)
        if os.path.normcase(os.path.realpath(kernel_python)) != os.path.normcase(os.path.realpath(sys.executable)):
            log.info("Note: the notebooks run in a different Python than this script; "
                     "the packages needed were found there, so the run continues.")

    t_all = time.time()
    done = []
    for n, (sid, rel) in enumerate(steps, 1):
        path = ROOT / rel
        log.info("-" * 78)
        log.info("[%d/%d] step %s: %s", n, len(steps), sid, rel)
        t0 = time.time()
        try:
            if path.suffix == ".ipynb":
                run_notebook(path, sid, kernel=args.kernel)
            elif path.suffix == ".Rmd":
                run_rmarkdown(path, rscript)
            else:
                run_rscript(path, rscript)
        except KeyboardInterrupt:
            log.info("Interrupted during step %s.", sid)
            return 130
        except Exception as err:  # noqa: BLE001 - report any failure and stop
            log.info("FAILED step %s after %s", sid, fmt_duration(time.time() - t0))
            log.info("%s", err)
            if done:
                log.info("Completed before the error: %s", ", ".join(done))
            log.info("Fix the problem and continue with:  python run_pipeline.py --from %s", sid)
            return 1
        done.append(sid)
        log.info("done step %s in %s", sid, fmt_duration(time.time() - t0))

    log.info("-" * 78)
    log.info("All %d step(s) finished in %s.", len(steps), fmt_duration(time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
