Meta-Learning for Adaptive Wireless Channel Estimation

This repository implements a meta-learning framework designed to address latency bottlenecks in OFDM wireless channel estimation. By utilizing First-Order MAML (FOMAML) and Reptile optimization algorithms, the framework enables a lightweight neural network to learn a generalized initialization that adapts rapidly to new wireless environments using a minimal number of pilot signals.

Key Achievements

Rapid Domain Adaptation: Achieved an optimal Normalized Mean Squared Error (NMSE) of 0.024 (-16.2 dB) using only 30 support samples for adaptation.

Performance: Significantly outperformed the standard train-from-scratch baseline (0.11 NMSE) in the few-shot regime.

Physics-Based Simulation: Engineered a custom data generation engine modeling 120 unique transmission environments with randomized multipath fading (2-5 paths) and Signal-to-Noise Ratios (5-25 dB).

Technical Overview

Traditional neural networks for channel estimation require large datasets to retrain for new environments, causing significant latency and bandwidth overhead. This project employs meta-learning to find a "meta-initialization"—a mathematically optimal starting state that allows the model to adapt to new environments with negligible gradient steps.

First-Order MAML (FOMAML)

The implementation utilizes First-Order MAML to optimize the initialization parameters. By approximating the meta-gradient and omitting the computationally expensive second-order Hessian matrix, the framework maintains high adaptation accuracy while reducing VRAM usage and training time by approximately 30%, making it suitable for edge-computing applications.

Channel Simulation Engine

The framework includes a custom simulator (generate_data.py) to create synthetic environments based on the OFDM channel model ($Y = XH + \text{noise}$).

The simulator generates 120 unique physical environments to ensure the model learns generalized channel physics rather than memorizing specific patterns. This includes:

Multipath Fading: Simulating phase shifts and distortion via randomized path profiles (2-5 paths).

Dynamic Noise Profiles: Modeling varying Signal-to-Noise Ratios (5-25 dB) to simulate diverse interference levels.

Performance Results

The following table summarizes the few-shot adaptation performance on unseen test environments:

Method

Adaptation Shots

Final NMSE

NMSE (dB)

Baseline (Train from scratch)

30

0.110

-9.5 dB

Reptile

30

0.035

-14.5 dB

FOMAML (Ours)

30

0.024

-16.2 dB

Quick Start Guide

Prerequisites

Install the required dependencies:

pip install torch numpy matplotlib


1. Generate Dataset

Generate the training and testing tasks. Data is saved as compressed .npz binaries in the results/ directory.

python generate_data.py --seed 42


2. Meta-Training

Train the 3-layer neural network using the meta-optimization loops. The output weights are saved as maml_model.pt.

python train.py


3. Evaluation

Run the test suite to evaluate few-shot adaptation on the held-out test environments and generate performance plots.

python test.py


Repository Structure

generate_data.py: Physics-based simulator for multipath fading and AWGN noise.

train.py: Implementation of FOMAML and Reptile meta-training loops.

test.py: Evaluation suite for adaptation performance and graph generation.

results/: Directory for datasets, trained model weights, and performance plots.

