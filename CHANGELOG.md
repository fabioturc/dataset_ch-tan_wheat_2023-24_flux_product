# Changelog

## v0.9.0 (2026-10-13)

Code version that produced flux product version **FP2026.10** (`dataset_CH-TAN_Nov2023-Aug2024_FP2026.10`), published in the ETH Research Collection record [10.3929/ethz-c-000807457](https://doi.org/10.3929/ethz-c-000807457) together with the [ley 2024-25 product](https://github.com/fabioturc/dataset_ch-tan_ley_2024-25_flux_product) and the raw input data. Pre-publication release: v1.0.0 will follow with the related article.

- Fixed: VPD is passed to REddyProc in hPa (diive gives kPa) in steps 51 and 81.5. This changes the MDS gap-filling of LE and H and the daytime partitioning (`GPP_DT`, `RECO_DT`).
- Fixed: the treatment variable `trt` is 0.5 for all records with uncertain parcel attribution (uncertain records of parcel B had 1). This changes the whole-footprint gap-filling only.
- Fixed random seeds in the R steps (bootstrap of the u* thresholds in 51, daytime partitioning in 81.5), so a re-run reproduces the product.
- New: `run_pipeline.py` runs all steps in order and logs the progress (see README).
- Notebook 90.0 also writes the full product as csv.
- The data are published in the ETH Research Collection (previously planned on Zenodo).
- Notebooks are committed without outputs (nbstripout, see `.gitattributes`).
