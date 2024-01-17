#! /bin/bash

cd trojan_detection

# train
python example_submission.py --mode val

python example_submission.py --mode val --verbose
# [PEZ] Combined Score: 0.165 Recall: 0.107 REASR: 0.222

python example_submission.py --mode test

# submission
cd submission && zip ../submission.zip ./* && cd ..

