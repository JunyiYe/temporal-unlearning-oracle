# P0 gate readout (tuning selection)

## transformer
- winner d_model=64 dropout=0.1 mean_val=0.50361 (seeds: [0.5015, 0.5045, 0.5048])
- runner-up d128/dr0.1 mean_val=0.50377; margin=0.00017 vs seed-spread=0.00180 (NEAR-TIE (either acceptable))
- all configs trained (epochs 3..300, none stuck at 1): [21, 22, 23, 24, 25]...
- emitted transformer_sp500_vol_v2.yaml

## itransformer
- winner d_model=32 dropout=0.1 mean_val=0.60170 (seeds: [0.6014, 0.6017, 0.602])
- runner-up d32/dr0.3 mean_val=0.60175; margin=0.00005 vs seed-spread=0.00039 (NEAR-TIE (either acceptable))
- all configs trained (epochs 3..300, none stuck at 1): [21, 22, 24, 25, 26]...
- emitted itransformer_sp500_vol_v2.yaml

## segrnn
- winner d_model=64 dropout=0.3 mean_val=0.52078 (seeds: [0.5194, 0.5209, 0.522])
- runner-up d32/dr0.1 mean_val=0.52124; margin=0.00046 vs seed-spread=0.00197 (NEAR-TIE (either acceptable))
- all configs trained (epochs 3..300, none stuck at 1): [37, 39, 40, 44, 45]...
- emitted segrnn_sp500_vol_v2.yaml

## patchtst
- winner d_model=32 dropout=0.1 mean_val=0.60055 (seeds: [0.6003, 0.6006, 0.6007])
- runner-up d32/dr0.3 mean_val=0.60057; margin=0.00002 vs seed-spread=0.00097 (NEAR-TIE (either acceptable))
- all configs trained (epochs 3..300, none stuck at 1): [24, 25, 26, 27, 29]...
- emitted patchtst_sp500_vol_v2.yaml

GATE P0: PASS
