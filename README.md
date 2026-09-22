# RAMD from Scratch

A step-by-step C++ learning project inspired by Random Acceleration Molecular Dynamics (RAMD).

## Goal

The purpose of this repository is to build a simplified RAMD-style particle model from basic C++ and molecular-dynamics concepts, while documenting the reasoning behind each step.

The project grows incrementally:

- **v0.1** — One-dimensional particle motion under a constant force
- **v0.2** — Multi-step trajectory generation with a `for` loop
- **Next** — Ligand center of mass, displacement thresholds, random direction updates, and repeated escape simulations

## Physical idea

The first versions use the basic chain:

```text
force -> acceleration -> velocity -> position
```

with

```text
a = F / m
v_new = v_old + a * dt
x_new = x_old + v_new * dt
```

Later versions will add simplified RAMD logic: monitor ligand center-of-mass displacement and change the applied direction when progress is too small.

## Build

```bash
g++ ramd_v01.cpp -o ramd_v01
./ramd_v01
```

or

```bash
g++ ramd_v02.cpp -o ramd_v02
./ramd_v02
```

## Learning focus

- variables -> store physical quantities
- arithmetic expressions -> implement equations
- loops -> advance simulation timesteps
- conditionals -> make RAMD direction-update decisions
- later: vectors, functions, random numbers, trajectory output, and statistics

## Reference

Inspired by Yiwei Ding, Alessia Ghidini, Pilar Cossio, and Benoit Roux, *A theoretical framework for random acceleration molecular dynamics simulations* (Journal of Chemical Physics, 2026).

This repository is an educational simplified implementation, not a reproduction of NAMD or a production molecular-dynamics engine.
