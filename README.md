# CommonLit Readability Prize
4th place solution code for the CommonLit Readability prize hosted on Kaggle (August 2021) https://www.kaggle.com/c/commonlitreadabilityprize

The writeup can be found here: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258148

# Setup
Edit `src/config.py` to reflect the input and output locations on your machine

## Apple Silicon (MPS) setup
For native Apple Silicon (M1/M2/M3), use the modern dependency set:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-modern.txt
```
Training auto-selects `mps` when available.
For OOF inference, `infer.py` now supports automatic device detection via:
```
python infer.py --timestamp <run_timestamp> --seed 48 --device auto
```

## Install as a package (pip)
You can install this repo as a local package and use the prediction CLI:
```
python -m pip install -e .
commonlit-predict "hello world" --timestamp <run_timestamp> --device auto
```

For batch scoring:
```
commonlit-predict --csv test.csv --timestamp <run_timestamp> --output test_predictions.csv
```

Note: model checkpoints/tokenizer files are loaded from `output/<timestamp>/<model_name>/fold_*/*.ckpt`.

## Offline delivery bundle (no internet required on target machine)
Build a self-contained offline bundle (package wheel, dependency wheels, and model artifacts):
```
source .venv/bin/activate
./scripts/build_offline_bundle.sh <run_timestamp> [model_name]
```

Example:
```
./scripts/build_offline_bundle.sh 20260404-172000 roberta-base
```

This generates:
- `offline_bundle/commonlit-offline-<timestamp>-<buildstamp>/`
- `offline_bundle/commonlit-offline-<timestamp>-<buildstamp>.tar.gz`

On the offline target machine:
```
tar -xzf commonlit-offline-<timestamp>-<buildstamp>.tar.gz
cd commonlit-offline-<timestamp>-<buildstamp>
python3 -m venv .venv
source .venv/bin/activate
./install_offline.sh
```

Then run:
```
export COMMONLIT_OUTPUT_PATH="$(pwd)/model"
commonlit-predict "hello world" --timestamp latest --device cpu
```

# Training
To train a single model using a config listed in `hyperparams.yml` run:
```
python train.py --config <config_name>
```
To run a 5-fold cross validation, using 5 different seeds, use the shell script `train.sh`. This script will also run `infer.py` and 
generate out-of-fold (OOF) predictions for stacking models.
```
sh train.sh <config_name>
```

# Inference
The final submission code that was used for inference in a Kaggle notebook is in the `submissions` folder
* `submission.py` - Public RMSE: 0.451, Private RMSE: 0.447
* `submission_netflix.py` - Public RMSE: 0.452, Private RMSE: 0.446