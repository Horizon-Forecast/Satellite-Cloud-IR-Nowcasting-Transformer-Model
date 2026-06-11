@echo off
REM RunPod Migration — compress training data
REM Run from project root: G:\dev\Horizon Forecast\Phase B\
REM Output: two .tar.zst archives ready to upload

set ROOT=G:\dev\Horizon Forecast\Phase B
set OUT=G:\runpod_upload

mkdir %OUT% 2>nul

echo ============================================
echo Step 1: Compress sat + ims + metadata
echo Estimated size: ~85-90 GB compressed
echo ============================================
tar --zstd -cf "%OUT%\horizon_sat.tar.zst" ^
    -C "%ROOT%\data\processed" sat_npy ^
    -C "%ROOT%\data\processed" ims_snapshots ^
    -C "%ROOT%\data\processed" index_train.csv ^
    -C "%ROOT%\data\processed" index_train_subset.csv ^
    -C "%ROOT%\data\processed" index_val.csv ^
    -C "%ROOT%\data\processed" norm_stats.json ^
    -C "%ROOT%\data\processed" dem_256.npy ^
    -C "%ROOT%\data\processed" station_mask.pt ^
    -C "%ROOT%\data\processed" rain_weights.pt
echo DONE: horizon_sat.tar.zst

echo ============================================
echo Step 2: Compress ERA5 (optional - can re-process on pod)
echo Estimated size: ~80-85 GB compressed
echo ============================================
tar --zstd -cf "%OUT%\horizon_era5.tar.zst" ^
    -C "%ROOT%\data" era5_npy
echo DONE: horizon_era5.tar.zst

echo ============================================
echo All done. Upload files from %OUT%\
echo ============================================
pause
