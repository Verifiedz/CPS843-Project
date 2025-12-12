# CPS843 Dev1 

## Install
pip install -r requirements.txt

## Run (single image)
python dcp.py --input path/to/hazy.jpg --output outputs/

## Run (folder)
python dcp.py --input path/to/hazy_folder --output outputs/

## Optional parameters
--patch-size 15
--omega 0.95
--t0 0.1
--no-refine
--no-clahe

## Quick sanity check
python dcp.py --self-test --input x --output y
