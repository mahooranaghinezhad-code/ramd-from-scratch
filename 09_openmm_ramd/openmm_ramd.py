"""
Paper 8 — OpenMM 7: rapid development of high-performance MD algorithms
(Eastman et al., PLoS Computational Biology 2017).

Minimal molecular-engine demonstration of a *RAMD-like* protocol.
Three connected particles represent a toy ligand; a custom centroid force
applies a 3D vector force to their mass-weighted center of mass.

NOT an all-atom protein-ligand simulation, not validated tauRAMD kinetics.
The well, parameters, and escape radius are synthetic teaching choices.
OpenMM provides the dynamics and thermostat; Python implements the external
direction-update protocol. All lengths are nm, time ps, energy kJ/mol.
Requires: pip install openmm numpy

Run:
 python 09_openmm_ramd/openmm_ramd.py
 python 09_openmm_ramd/openmm_ramd.py --replicas 12 --force 50 --seed 2026

Outputs:
 results_paper8/trajectory.csv
 results_paper8/escape_summary.csv
 results_paper8/first_replica.dcd (optional via DCDReporter)

Unlike Paper 1's exponential direction persistence model, this code
implements a displacement-threshold direction update, similar in *logic*
to tauRAMD (Paper 10), without claiming matching atomistic parameters.
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from openmm import (
    System, HarmonicBondForce, CustomCentroidBondForce,
    LangevinMiddleIntegrator, Platform, Vec3, unit,
)
from openmm.app import Simulation, Topology, DCDReporter, element


def unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def make_model(force_magnitude, well_depth, well_width):
    masses = [12.0, 16.0, 12.0]  # dalton
    system = System()
    for m in masses:
        system.addParticle(m * unit.dalton)

    # Three bonded atoms, representing a small deformable ligand.
    bonds = HarmonicBondForce()
    bonds.addBond(0, 1, 0.12 * unit.nanometer,
                  20000 * unit.kilojoule_per_mole / unit.nanometer**2)
    bonds.addBond(1, 2, 0.12 * unit.nanometer,
                  20000 * unit.kilojoule_per_mole / unit.nanometer**2)
    bonds.addBond(0, 2, 0.12 * unit.nanometer,
                  20000 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bonds)

    # Two forces act on the same COM group:
    # external RAMD vector and a stationary smooth attractive well.
    expression = (
        "-F*(nx*x1 + ny*y1 + nz*z1)"
        "-depth*exp(-(x1*x1+y1*y1+z1*z1)/(2*width*width))"
    )
    centroid = CustomCentroidBondForce(1, expression)
    centroid.addGlobalParameter("F", force_magnitude)
    centroid.addGlobalParameter("nx", 1.0)
    centroid.addGlobalParameter("ny", 0.0)
    centroid.addGlobalParameter("nz", 0.0)
    centroid.addGlobalParameter("depth", well_depth)
    centroid.addGlobalParameter("width", well_width)
    group = centroid.addGroup([0, 1, 2], masses)
    centroid.addBond([group], [])
    system.addForce(centroid)

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("TOY", chain)
    atoms = [
        topology.addAtom(f"C{i+1}", element.carbon, residue)
        for i in range(3)
    ]
    topology.addBond(atoms[0], atoms[1])
    topology.addBond(atoms[1], atoms[2])
    topology.addBond(atoms[0], atoms[2])
    return topology, system, np.array(masses)


def center_of_mass(simulation, masses):
    state = simulation.context.getState(getPositions=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    com = np.average(np.asarray(coordinates), axis=0, weights=masses)
    return com, np.asarray(coordinates)


def set_direction(simulation, vector):
    for key, value in zip(("nx", "ny", "nz"), vector):
        simulation.context.setParameter(key, float(value))


def simulate(args):
    if args.replicas <= 0 or args.check_steps <= 0 or args.steps <= 0:
        raise ValueError("Simulation counts must be positive")
    if args.dt <= 0 or args.force < 0 or args.radius <= 0:
        raise ValueError("dt, radius must be positive and force non-negative")
    if args.depth < 0 or args.width <= 0 or args.threshold < 0:
        raise ValueError("Invalid well or threshold settings")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    topology, system, masses = make_model(
        args.force, args.depth, args.width
    )
    rng = np.random.default_rng(args.seed)
    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.dt * unit.picosecond,
    )
    platform = Platform.getPlatformByName(args.platform)
    simulation = Simulation(topology, system, integrator, platform)

    initial = [
        Vec3(-0.06, -0.03464, 0.0),
        Vec3(0.06, -0.03464, 0.0),
        Vec3(0.0, 0.06928, 0.0),
    ] * unit.nanometer

    trajectory_path = output / "trajectory.csv"
    summary_path = output / "escape_summary.csv"
    with open(trajectory_path, "w", newline="", encoding="utf-8") as tf, \
         open(summary_path, "w", newline="", encoding="utf-8") as sf:
        tw, sw = csv.writer(tf), csv.writer(sf)
        tw.writerow([
            "replica", "step", "time_ps", "com_x_nm", "com_y_nm",
            "com_z_nm", "com_radius_nm", "dir_x", "dir_y", "dir_z",
            "direction_changed",
        ])
        sw.writerow([
            "replica", "seed", "escaped", "escape_time_ps",
            "direction_changes", "observation_time_ps",
        ])

        for replica in range(args.replicas):
            seed = int(rng.integers(1, 2**31 - 1))
            integrator.setRandomNumberSeed(seed)
            simulation.context.setPositions(initial)
            simulation.context.setVelocitiesToTemperature(
                args.temperature * unit.kelvin, seed
            )
            # Context parameters persist between replicas: reset them.
            direction = unit_vector(rng)
            set_direction(simulation, direction)

            if replica == 0 and args.save_dcd:
                simulation.reporters.append(
                    DCDReporter(str(output / "first_replica.dcd"),
                                args.record_steps)
                )
            last_check_com, _ = center_of_mass(simulation, masses)
            changes = 0
            escaped = False
            escape_time = None
            step_count = 0
            while step_count < args.steps:
                chunk = min(args.check_steps, args.steps - step_count)
                simulation.step(chunk)
                step_count += chunk

                com, _ = center_of_mass(simulation, masses)
                radius = float(np.linalg.norm(com))
                changed = False
                if radius >= args.radius:
                    escaped = True
                    # First crossing was somewhere within this chunk.
                    escape_time = step_count * args.dt
                elif np.linalg.norm(com - last_check_com) < args.threshold:
                    direction = unit_vector(rng)
                    set_direction(simulation, direction)
                    changes += 1
                    changed = True
                last_check_com = com.copy()

                tw.writerow([
                    replica, step_count, step_count * args.dt,
                    *[float(z) for z in com], radius,
                    *[float(z) for z in direction], int(changed),
                ])
                if escaped:
                    break

            sw.writerow([
                replica, seed, int(escaped),
                "" if escape_time is None else escape_time,
                changes, step_count * args.dt,
            ])
            if replica == 0:
                simulation.reporters.clear()
            print(f"replica {replica}: escaped={escaped}, "
                  f"changes={changes}, observed={step_count * args.dt:.2f} ps")

    print("Saved:", trajectory_path, summary_path)
    print("Caution: no protein, solvent, calibrated force or residence-time estimate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_paper8")
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--check-steps", type=int, default=50)
    parser.add_argument("--record-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--force", type=float, default=50)
    parser.add_argument("--depth", type=float, default=10)
    parser.add_argument("--width", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.0025)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--platform", choices=["Reference", "CPU", "CUDA", "OpenCL"], default="Reference")
    parser.add_argument("--save-dcd", action="store_true")
    simulate(parser.parse_args())


if __name__ == "__main__":
    main()
