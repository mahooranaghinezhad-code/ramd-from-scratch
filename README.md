# RAMD from Scratch

A step-by-step C++ learning project inspired by Random Acceleration Molecular Dynamics (RAMD).

## Goal

The purpose of this repository is to build a simplified RAMD-style particle model from basic C++ and molecular-dynamics concepts, while documenting the reasoning behind each step.

The project grows incrementally:

- **v0.1** - One-dimensional particle motion under a constant force
- **v0.2** - Multi-step trajectory generation with a `for` loop
- **v0.3 concept** - Ligand center of mass (COM)
- **v0.4** - Integrated ligand motion, COM displacement, and threshold decision logic
- **Next** - Random force-direction changes and repeated escape simulations

## Physical idea

The first versions use the chain:

```text
force -> acceleration -> velocity -> position
```

with

```text
a = F / m
v_new = v_old + a * dt
x_new = x_old + v_new * dt
```

The integrated version treats the ligand as a simplified translating three-atom object. It calculates the ligand center of mass, moves the ligand, measures COM displacement, and checks whether the displacement is below a threshold.

```text
atoms -> COM -> force -> acceleration -> motion
                         |
                         v
               new COM -> displacement
                              |
                              v
                  displacement < threshold?
```

The next version will replace the placeholder "change direction needed" decision with an actual random direction update.

## Build

For example:

```bash
g++ ramd_v04_integrated.cpp -o ramd_v04
./ramd_v04
```

## Learning focus

- variables -> store physical quantities
- arithmetic expressions -> implement equations
- loops -> advance simulation timesteps
- center of mass -> represent whole-ligand translation
- conditionals -> make RAMD direction-update decisions
- later: vectors, functions, random numbers, trajectory output, and statistics

## Reference

Inspired by Yiwei Ding, Alessia Ghidini, Pilar Cossio, and Benoit Roux, *A theoretical framework for random acceleration molecular dynamics simulations* (Journal of Chemical Physics, 2026).

This repository is an educational simplified implementation, not a reproduction of NAMD or a production molecular-dynamics engine.
