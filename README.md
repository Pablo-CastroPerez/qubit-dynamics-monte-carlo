# Quantum Dynamics Simulation of a Two-Level Atom

This repository contains a simulation of a **two-level quantum system** (qubit) driven by a **resonant laser field**, incorporating **spontaneous emission** via the **Monte Carlo** method.

The code compares the **analytical solution** for excited-state population \( |b|^2 \) with **stochastic simulations** over serveral trajectories (qubits).

---

## Physics Overview

- **System**: Two-level atom/qubit under resonant drive  
- **Method**: Non-Hermitian Schrödinger evolution + quantum jumps  
- **Observable**: Excited-state population \( |b(t)|^2 \)  
- **Tools**: Python, NumPy, Matplotlib

For theory and simulation results, see the 📄 [poster](Qubit_Resonance_Poster.pdf).

---

## Files

| File                     | Description                               |
|--------------------------|-------------------------------------------|
| `Qubit_Resonance.py`     | Core simulation and plotting code         |
| `Qubit_Resonance_Poster.pdf` | Summary poster with theory and results  |

---

## Example Output

The script generates a plot comparing:
- Analytical solution
- Monte Carlo averages (1,000 and 10,000 qubits)

---
