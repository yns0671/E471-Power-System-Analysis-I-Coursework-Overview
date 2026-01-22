# EE471 Term Project – Fast Decoupled Load Flow (FDLF)

This repository contains the Python implementation of a **generic Fast Decoupled Load Flow (FDLF) solver** developed for the course **EE-471 Power System Analysis I (2025–2026 Fall Term)**.

The solver is designed to work with **arbitrary power system networks** described using JSON-based topology and load-flow input files. It computes steady-state bus voltages and total system losses efficiently, even for large-scale systems.

## Project Overview

- Method: Fast Decoupled Load Flow (FDLF)
- Language: Python 3.13+
- Input Format: JSON (topology and load-flow case files)
- Output:
  - Bus voltage magnitudes (p.u.)
  - Bus voltage angles (radians)
  - Total active power loss (MW)
  - Total reactive power loss (MVAR)

## Features

- Generic implementation (no hard-coded bus numbers)
- Dynamic bus ID mapping
- Sparse Y-bus construction
- Efficient solution using decoupled matrices
- Compatible with IEEE test systems (14, 30, 57, 118, 300 bus)

## Validation

The solver results were validated against:
- **PSSE** (Newton–Raphson power flow)
- **DIgSILENT PowerFactory** (for IEEE 14 and 30 bus systems)

Voltage magnitude errors were consistently below **0.3%**, and convergence behavior matched expected theoretical performance of the FDLF method.

## Usage

1. Prepare topology and load-flow JSON files according to the provided data format.
2. Run the main Python script.
3. The solver outputs voltage results, convergence history, and system losses.

## Notes

- Detailed explanations of the algorithm are provided in the **project report**, not in this README.

## Author

**Yunus Tosun**  
EE-471 Power System Analysis I  
2025–2026 Fall Term
