# Rare-Event VRU Motion Forecasting

> **Trajectory forecasting for pedestrians, cyclists, and motorcyclists, with a focus on rare and safety-critical motion patterns.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Argoverse%202-6f42c1.svg)](https://www.argoverse.org/av2.html)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#project-status)

## Overview

Most motion-forecasting models are optimized for average performance. This can hide poor behavior on the rare trajectories that matter most in safety-critical driving scenarios: abrupt turns, sudden acceleration, sharp braking, unusual lane changes, and aggressive cyclist or motorcyclist maneuvers.

This project studies **rare-event tail forecasting for vulnerable road users (VRUs)** using the Argoverse 2 Motion Forecasting Dataset. The current pipeline:

1. extracts pedestrian, cyclist, and motorcyclist focal agents;
2. transforms trajectories into a local agent-centric coordinate frame;
3. derives motion statistics such as speed, acceleration, and heading change;
4. labels statistically rare trajectories within each VRU class;
5. trains GRU-based trajectory-forecasting baselines;
6. evaluates normal and tail trajectories separately;
7. investigates a multi-task tail-aware forecasting model.

The central research question is:

> **Can tail-event performance be improved without degrading average forecasting accuracy?**

---

## Motivation

Average Displacement Error (ADE) and Final Displacement Error (FDE) are useful global metrics, but they are dominated by frequent and relatively predictable trajectories. In real road-safety applications, a forecasting system must also handle low-frequency, high-risk behavior.

Cyclists and motorcyclists are particularly important because they:

- have more variable dynamics than pedestrians;
- can accelerate and change direction rapidly;
- are underrepresented in many public datasets;
- are highly exposed in collisions;
- may perform uncommon but safety-critical maneuvers near intersections.

A model with good average ADE may still fail badly on exactly these cases.

---

## Dataset

The project uses the **Argoverse 2 Motion Forecasting Dataset**.

Each trajectory contains:

- **50 observed steps**: 5 seconds of motion history;
- **60 future steps**: 6 seconds to predict;
- sampling frequency: **10 Hz**.

Only focal agents belonging to the following classes are retained:

- `pedestrian`
- `cyclist`
- `motorcyclist`

### VRU subset statistics

| Split | Pedestrian | Cyclist | Motorcyclist | Total |
|---|---:|---:|---:|---:|
| Train | 12,845 | 2,692 | 1,038 | **16,575** |
| Validation | 1,572 | 303 | 134 | **2,009** |

The distribution is strongly imbalanced, with pedestrians representing most available trajectories.

> The Argoverse 2 dataset is **not included** in this repository. Users must download it separately and follow the official Argoverse licensing terms.

---

## Trajectory preprocessing

Raw global coordinates are converted to an agent-centric frame.

For every trajectory:

1. the last observed position is translated to the origin;
2. all positions are expressed relative to that point;
3. the trajectory is rotated so that the latest observed heading aligns with the positive x-axis.

The resulting tensors are:

```text
past:   (50, 2)
future: (60, 2)
```

This reduces irrelevant variation caused by absolute map position and global orientation.

---

## Tail-event definition

A **tail event** is a statistically rare motion pattern within the same road-user class.

For each trajectory, the pipeline computes:

- mean and maximum speed;
- mean and maximum acceleration;
- mean and maximum heading change;
- total displacement.

A preliminary tail score is defined as:

```text
tail_score = 0.4 × normalized_max_speed
           + 0.4 × normalized_max_acceleration
           + 0.2 × normalized_max_heading_change
```

The top 5% of trajectories are labeled as tail events.

### Why class-wise labeling?

A global threshold is biased because the natural dynamics of each class differ. A normal motorcyclist is typically faster than a normal pedestrian. Under one shared threshold, many ordinary motorcyclist trajectories were incorrectly classified as rare.

The current approach therefore normalizes features and applies the 95th-percentile threshold **separately for each class**.

### Class-wise training labels

| Class | Normal | Tail | Tail rate |
|---|---:|---:|---:|
| Pedestrian | 12,202 | 643 | 5.01% |
| Cyclist | 2,557 | 135 | 5.01% |
| Motorcyclist | 986 | 52 | 5.01% |
| **Total** | **15,745** | **830** | **5.01%** |

### Validation labels

| Class | Normal | Tail |
|---|---:|---:|
| Pedestrian | 1,493 | 79 |
| Cyclist | 287 | 16 |
| Motorcyclist | 127 | 7 |
| **Total** | **1,907** | **102** |

> **Important methodological note:** the current prototype computed validation scaling and thresholds independently on the validation split. For publication-quality evaluation, all normalization parameters and thresholds must be estimated on the training split only and then applied unchanged to validation and test data.

---

## Models

### 1. Kinematic baselines

Two non-learning baselines were implemented:

- **Constant Velocity**
- **Constant Acceleration**

### 2. GRU trajectory baseline

The baseline neural predictor uses:

- a 2-layer GRU encoder;
- hidden dimension: 128;
- a fully connected decoder;
- output: 60 future 2D positions.

### 3. Tail-Aware Multi-Task GRU

The tail-aware model uses one shared GRU encoder and two heads:

```text
Observed trajectory
        │
        ▼
   Shared GRU encoder
      ┌───────┴────────┐
      ▼                ▼
Trajectory head   Tail classifier
  60 × 2 points    normal / tail
```

The training objective is:

```text
L = L_trajectory + λ L_tail
```

The tail classifier is an auxiliary task intended to encourage the encoder to learn representations associated with difficult motion patterns.

---

## Evaluation metrics

### Average Displacement Error

```text
ADE = mean Euclidean distance between predicted and ground-truth future positions
```

### Final Displacement Error

```text
FDE = Euclidean distance at the final prediction step
```

Results are reported for:

- all validation trajectories;
- normal trajectories;
- tail trajectories;
- each VRU class separately.

Tail classification is evaluated using precision, recall, F1-score, and a confusion matrix. Accuracy alone is not considered sufficient because tail events represent only about 5% of the data.

---

## Current results

### Kinematic baselines

| Model | ADE (m) | FDE (m) |
|---|---:|---:|
| Constant Velocity | 4.4022 | 11.6556 |
| Constant Acceleration | 9.5523 | 28.5941 |

### GRU baseline

| Subset | ADE (m) | FDE (m) | Samples |
|---|---:|---:|---:|
| Overall | **1.2353** | **2.7232** | 2,009 |
| Normal | **1.1569** | **2.5283** | 1,907 |
| Tail | **2.7004** | **6.3666** | 102 |

Tail trajectories have approximately **2.5× larger FDE** than normal trajectories.

### Baseline results by class

| Class | Overall ADE | Overall FDE | Tail ADE | Tail FDE | Tail samples |
|---|---:|---:|---:|---:|---:|
| Pedestrian | 0.8838 | 1.8366 | 1.9552 | 4.3950 | 79 |
| Cyclist | 1.9071 | 4.2605 | 4.1668 | 9.8823 | 16 |
| Motorcyclist | 3.8395 | 9.6487 | 7.7591 | 20.5818 | 7 |

The motorcyclist tail subset is the most difficult, although the validation sample count is currently too small for strong statistical conclusions.

### Tail-Aware GRU v1

The first multi-task experiment used balanced sampling, changing the effective training distribution from approximately 95/5 to 50/50.

| Subset | ADE (m) | FDE (m) |
|---|---:|---:|
| Overall | 1.6131 | 3.5613 |
| Normal | 1.5318 | 3.3698 |
| Tail | 3.1325 | 7.1427 |

Tail classification:

| Metric | Value |
|---|---:|
| Accuracy | 0.8870 |
| Precision | 0.1981 |
| Recall | 0.4020 |
| F1-score | 0.2654 |

Confusion matrix:

```text
TP = 41
TN = 1741
FP = 166
FN = 61
```

The v1 model did **not** improve forecasting. Oversampling the limited tail set likely distorted the trajectory-learning distribution and produced negative transfer.

### Tail-Aware GRU v2 training

The second experiment:

- removes balanced sampling;
- preserves the original data distribution;
- uses weighted binary cross-entropy for the auxiliary classifier;
- reduces the auxiliary loss weight to `λ = 0.05`;
- selects checkpoints based on trajectory loss.

Best observed training trajectory loss:

```text
3.1569
```

Validation evaluation for v2 is the next experimental step.

---

## Preliminary findings

The project currently supports three main conclusions:

1. **A global tail-event threshold is biased across VRU classes.**  
   Class-wise labeling is necessary because pedestrians, cyclists, and motorcyclists have different natural motion distributions.

2. **Rare trajectories are substantially harder to predict.**  
   The GRU baseline reaches 2.5283 m normal FDE versus 6.3666 m tail FDE.

3. **Naive oversampling can degrade both average and tail performance.**  
   Tail awareness cannot be introduced by simply repeating rare samples until the training set appears balanced.

These are preliminary research findings, not final claims.

---

## Repository structure

The intended project structure is:

```text
rare-event-vru-motion-forecasting/
├── models/
│   ├── gru.py
│   └── tail_gru.py
├── scripts/
│   ├── create_classwise_tail_labels.py
│   ├── evaluate_tail_gru.py
│   ├── evaluate_tail_gru_v2.py
│   ├── train_tail_gru.py
│   └── train_tail_gru_v2.py
├── src/
│   ├── dataset.py
│   └── tail_dataset.py
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

Model checkpoints, generated CSV files, and the Argoverse 2 dataset should not be committed.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/panagiotagrosdouli/rare-event-vru-motion-forecasting.git
cd rare-event-vru-motion-forecasting
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are expected to include:

```text
torch
numpy
pandas
pyarrow
```

---

## Running the experiments

Create class-wise tail labels:

```bash
PYTHONPATH=. python scripts/create_classwise_tail_labels.py --split train
PYTHONPATH=. python scripts/create_classwise_tail_labels.py --split val
```

Train the first tail-aware model:

```bash
PYTHONPATH=. python scripts/train_tail_gru.py
```

Train the revised tail-aware model:

```bash
PYTHONPATH=. python scripts/train_tail_gru_v2.py
```

Evaluate a trained model:

```bash
PYTHONPATH=. python scripts/evaluate_tail_gru.py
```

Paths may need to be adjusted according to the local Argoverse 2 directory structure.

---

## Roadmap

- [x] Extract VRU focal-agent trajectories from Argoverse 2
- [x] Implement agent-centric trajectory normalization
- [x] Build constant-velocity and constant-acceleration baselines
- [x] Train a GRU forecasting baseline
- [x] Create class-wise statistical tail labels
- [x] Evaluate normal and tail subsets independently
- [x] Implement a multi-task tail-aware GRU
- [x] Diagnose failure caused by balanced oversampling
- [ ] Evaluate Tail-Aware GRU v2 on validation data
- [ ] Fit tail thresholds on training data only
- [ ] Add confidence intervals and repeated-seed experiments
- [ ] Perform ablation studies on tail features and loss weighting
- [ ] Add qualitative trajectory visualizations
- [ ] Investigate human-reviewed tail-event annotation
- [ ] Compare against stronger motion-forecasting baselines

---

## Research direction

The long-term goal is to build a curated rare-event benchmark for cyclists and motorcyclists and develop forecasting methods that improve safety-critical tail performance without sacrificing normal-case accuracy.

A motivating real-world case is a motorcycle approaching an intersection and suddenly overtaking a stopped or slowing vehicle. A forecasting system that predicts only the average motion pattern may continue the motorcycle along its previous path, while a tail-aware model should better capture the possibility of a rapid lateral maneuver.

Potential future extensions include:

- uncertainty-aware and multimodal forecasting;
- class-conditioned tail definitions;
- distributionally robust optimization;
- hard-example mining;
- focal or ranking-based objectives;
- human annotation of semantic maneuver types;
- evaluation on additional datasets.

---

## Project status

This repository is a **research prototype and work in progress**.

The reported numbers are useful for experimentation and diagnosis, but the benchmark is not yet publication-ready. In particular:

- validation tail thresholds must be derived from training statistics only;
- results should be repeated across multiple random seeds;
- the small number of motorcyclist tail samples requires careful interpretation;
- the current weighted tail score is a preliminary design choice;
- stronger baselines and qualitative analysis are still required.

---

## Acknowledgements

This project uses the Argoverse 2 Motion Forecasting Dataset. Please cite the official Argoverse 2 publications when using the dataset in academic work.

---

## Author

**Panagiota Grosdouli**

Research interests: autonomous driving, vulnerable road-user trajectory forecasting, intelligent transportation systems, and safety-critical machine learning.
