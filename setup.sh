#!/bin/bash
conda env create -f environment.yml
conda activate swint
pip install gdown
gdown https://drive.google.com/uc?id=1CC0DwkrJ3Lb-DhHXrmQ8g6mCr3I74umf
cp -r mmaction/* /srv/conda/envs/swint/lib/python3.7/site-packages/mmaction/
https://drive.google.com/file/d/1z6Wqx2y0rUD_YyAWiwEAD8dgvHvAAgaD/view?usp=sharing