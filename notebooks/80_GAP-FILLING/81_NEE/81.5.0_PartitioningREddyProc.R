# LOAD LIBRARIES----------------------------------------------------------------
library(tidyverse)
library(REddyProc)
library(caTools)
library(data.table)
library(viridis)
library(mlegp)

# LOAD DATA---------------------------------------------------------------------
file_fluxes_meteo <- "notebooks/80_GAP-FILLING/81_NEE/81.4.2_PartitioningSubsetForREddyProc.csv"
filedata <- read_csv(file_fluxes_meteo, show_col_types = FALSE)

# SETTINGS----------------------------------------------------------------------
output_path = 'notebooks/80_GAP-FILLING/81_NEE/'

# FUNCTION----------------------------------------------------------------------
XG_Partition <- function(data, date_start, date_end) {
  
  # record output
  logf <- file(paste0(output_path, "81.5.1_logfile_PartitioningNEE_XGf.txt"))
  sink(logf, type = "output", split = TRUE, append = TRUE)
  sink(logf, append=TRUE, type="message")
  
  # load data
  df_XG <- data %>% 
    select(
      TIMESTAMP_END, 
      USTAR,
      ppfd, 
      vpd, 
      sw_in,
      ta, 
      rh, 
      NEE_L3.3_CUT_50_QCF_footprint_gfXGBoost, 
      NEE_L3.3_CUT_50_QCF0_footprint_gfXGBoost
  )
  
  cat("Data selected \n")
  
  
  # Check timestamp
  cat("Check timestamp \n")
  print(head(df_XG$TIMESTAMP_END))
  
  # Convert timestamp to posix
  # CET clock times parsed as UTC by readr: relabel as CET without shifting the clock (see 51.0 Rmd)
  df_XG$TIMESTAMP_END <- lubridate::force_tz(as.POSIXct(df_XG$TIMESTAMP_END, format="%Y-%m-%d %H:%M:%S", tz = "UTC"), tzone = "Etc/GMT-1")
  
  assign("df_XG", df_XG, envir = .GlobalEnv)
  
  # stop and check using browser() 
  # type c in the console
  # Q is quite 
  
  # Prepare EddyProc data
  EddyData.F <- df_XG[FALSE]
  EddyData.F <- EddyData.F %>% 
    mutate(
    TIMESTAMP = df_XG$TIMESTAMP_END,
    NEE_QCF = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF_footprint_gfXGBoost)),
    NEE_QCF0 = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_footprint_gfXGBoost)),
    Ustar = as.numeric(as.character(df_XG$USTAR)),
    Rg = as.numeric(as.character(df_XG$sw_in)),
    Tair = as.numeric(as.character(df_XG$ta)),
    RH = as.numeric(as.character(df_XG$rh)),
    VPD = pmax(as.numeric(as.character(df_XG$vpd)), 0)
    ) %>% 
    filter(TIMESTAMP >= as.POSIXct(paste0(date_start, ' 00:30:00'), tz = attr(TIMESTAMP, 'tzone'))) %>%
    filter(TIMESTAMP <= as.POSIXct(paste0(date_end, ' 00:00:00'), tz = attr(TIMESTAMP, 'tzone'))) %>%
    mutate(TIMESTAMP_STRING = as.character(TIMESTAMP)) %>%
    mutate(TIMESTAMP_STRING = ifelse(nchar(TIMESTAMP_STRING) == 10, 
                                     paste0(TIMESTAMP_STRING, " 00:00:00"), 
                                     TIMESTAMP_STRING))
  
  cat("Input data check \n")
  print(str(EddyData.F))
  cat("Input data ready \n")
  assign("XGEddyData.F", EddyData.F, envir = .GlobalEnv)
  
  # REddyProc processing
  EddyProc.C <- sEddyProc$new('CH-TAN', EddyData.F,
                              c('NEE_QCF', 'NEE_QCF0',
                                'Rg', 'Tair', 'Ustar', 'VPD'),
                              ColPOSIXTime = "TIMESTAMP")
  EddyProc.C$sSetLocationInfo(LatDeg = 47.480620, LongDeg = 8.911868, TimeZoneHour = 1)
  
  # Check data
  cat("Check REddyProc timestamp \n")
  print(str(str(EddyProc.C)))
  print(head(EddyProc.C$sDATA$sDateTime))
  print(head(EddyProc.C$sTEMP)) # Middle timestamp
  print(head(EddyProc.C$sDATA))
  
  cat("REddyProc ready \n")
  cat("Start MDS Gapfilling \n")
  
  # MDS Gap-filling
  EddyProc.C$sMDSGapFill('Tair', FillAll = FALSE)
  EddyProc.C$sMDSGapFill('Rg', FillAll = FALSE)
  EddyProc.C$sMDSGapFill('VPD', FillAll = FALSE)
  cat("Meteo gapfilled \n")
  
  EddyProc.C$sMDSGapFill('NEE_QCF', FillAll = TRUE)
  cat("NEE QCF already gapfilled with XGBoost was gapfilled with MDS if any gaps still existing\n")
  
  EddyProc.C$sMDSGapFill('NEE_QCF0', FillAll = TRUE)
  cat("NEE QCF0 already gapfilled with XGBoost was gapfilled with MDS if any gaps still existing\n")
  
  # Export filled data
  FillEddyData.F <- EddyProc.C$sExportResults()
  FillEddyData.F$TIMESTAMP <- EddyData.F$TIMESTAMP_STRING

  assign("XGFillEddyData.F", FillEddyData.F, envir = .GlobalEnv)
  
  # Partitioning
  cat("Start partitioning \n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'QCF')
  EddyProc.C$sGLFluxPartition(suffix = 'QCF')
  cat("NEE QCF already gapfilled with XGBoost was partitioned\n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'QCF0')
  EddyProc.C$sGLFluxPartition(suffix = 'QCF0')
  cat("NEE QCF0 already gapfilled with XGBoost was partitioned\n")


  # Export partitioned data
  FillPartitionEddyData.F <- EddyProc.C$sExportResults()
  FillPartitionEddyData.F$TIMESTAMP <- EddyData.F$TIMESTAMP_STRING
  filename = "81.5.1_NEE_XG-GAPF_PART_ReddyProc.csv"
  write.csv(FillPartitionEddyData.F, file = paste0(output_path, filename), row.names = FALSE)
  cat(paste0("Files saved as: ", filename, "\n"))
  assign("XGFillPartitionEddyData.F", FillPartitionEddyData.F, envir = .GlobalEnv)

  # Save fingerprint plots
  plots_path = "notebooks/80_GAP-FILLING/81_NEE/plots"
  EddyProc.C$sPlotFingerprint('GPP_QCF_f', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_QCF', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('Reco_QCF', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_QCF', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('GPP_QCF0_f', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_QCF0', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('Reco_QCF0', Dir = plots_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_QCF0', Dir = plots_path)
  
  cat("Partitioning done \n")

}

# RUN FUNCTION------------------------------------------------------------------
# Dataset period (TIMESTAMP_END): 17 Oct 2023 00:30 to 23 Aug 2024 00:00; must match src/config.py
XG_Partition(filedata, '2023-10-17', '2024-08-23')

