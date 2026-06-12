# Trojan Detection Challenge 2023 (LLM Edition)

**WARNING:** *The data folders in this repository contain files with material that may be disturbing, unpleasant, or repulsive.*

This repository is based on the starter kit for the **Trojan Detection Challenge
2023 (LLM Edition)**, a NeurIPS 2023 competition. It contains the per-track
starter code together with this author's experiments, baseline runs, and
submission artifacts. To learn more about the competition, see the
[competition website](https://trojandetection.ai/).

## Tracks

Each track lives in its own folder and has its own README with detailed
instructions for downloading data, running baselines, and generating
submissions:

- **`trojan_detection/`** — Trojan Detection track. Includes `download_data.py`,
  baseline code under `baselines/`, `example_submission.py` /
  `example_submission.ipynb` (plus a Colab variant), `eval_utils.py`,
  `postprocess.py`, `run.sh`, computed `results.json`, and several packaged
  submission archives (`submission_v*.zip`).
- **`red_teaming/`** — Red Teaming track. Includes `download_data.py`, baseline
  code under `baselines/`, `example_submission.py` /
  `example_submission.ipynb`, and `eval_utils.py`.

## Getting Started

Each track is set up independently. From within a track folder:

```bash
pip install -r requirements.txt   # install track dependencies
python download_data.py           # download the track's data
bash run.sh                       # (trojan_detection) run the baseline pipeline
```

See the README inside `trojan_detection/` and `red_teaming/` for the full,
track-specific instructions, including how to format and package a submission.

## Dependencies

Python with the packages listed in each track's `requirements.txt` (PyTorch /
Hugging Face Transformers based, as used for working with the competition's
language models).
