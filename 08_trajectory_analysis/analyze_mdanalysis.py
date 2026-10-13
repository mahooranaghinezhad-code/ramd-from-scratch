"""Paper 7 — MDAnalysis trajectory analysis (Michaud-Agrawal et al., 2011).

Reads a real molecular topology (PDB/PSF etc.) and trajectory (DCD/XTC etc.)
and extracts COM distance, ligand displacement, ligand-to-receptor contacts,
protein-aligned ligand RMSD, and a first-crossing escape proxy.

This is a pedagogical analysis pipeline; a distance cutoff is not proof of
irreversible ligand dissociation. Validate PBC, topology, alignment, sampling
intervals and state definitions before scientific interpretation.

Example:
 python 08_trajectory_analysis/analyze_mdanalysis.py \
   --topology complex.pdb --trajectory production.dcd \
   --protein "protein" --ligand "resname LIG" --escape-distance 30
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import distance_array


def calculate_contact_count(a, b, cutoff, box):
    """Count atom pairs separated by less than cutoff (Å in MDAnalysis units)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    distances = distance_array(a.positions, b.positions, box=box)
    return int(np.count_nonzero(distances < cutoff))


def get_box(universe, use_pbc):
    if not use_pbc:
        return None
    dims = universe.dimensions
    if dims is None or np.any(np.asarray(dims[:3]) <= 0):
        raise ValueError("PBC requested but trajectory has invalid unit-cell dimensions.")
    return dims


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--protein", default="protein")
    ap.add_argument("--ligand", required=True, help='Example: "resname LIG"')
    ap.add_argument("--align", default="protein and backbone",
                    help="Stable receptor atoms for best-fit alignment")
    ap.add_argument("--contact-cutoff", type=float, default=4.5)
    ap.add_argument("--escape-distance", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--use-pbc", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results_paper7"))
    args = ap.parse_args()
    if args.stride < 1 or args.contact_cutoff <= 0 or args.escape_distance <= 0:
        ap.error("stride and cutoffs must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(args.topology, args.trajectory)
    protein = u.select_atoms(args.protein)
    ligand = u.select_atoms(args.ligand)
    anchor = u.select_atoms(args.align)
    if not len(protein) or not len(ligand) or len(anchor) < 3:
        raise ValueError("Selection empty or alignment selection too small; check atom names.")
    if len(protein) > 15000 or len(ligand) > 400:
        print("Warning: contact calculation may be expensive for this selection.")

    # Alignment is essential for meaningful ligand RMSD with respect to the
    # receptor. Reference is the first analyzed frame, not an absolute binding pose.
    u.trajectory[0]
    reference = mda.Universe(args.topology, args.trajectory)
    reference.trajectory[0]
    ref_anchor = reference.select_atoms(args.align)
    ref_ligand = reference.select_atoms(args.ligand)
    if len(anchor) != len(ref_anchor) or len(ligand) != len(ref_ligand):
        raise ValueError("Reference atom order/selection mismatch.")

    reference_ligand_coords = ref_ligand.positions.copy()
    ref_ligand_com = ref_ligand.center_of_mass()
    records = []
    initial_ligand_com = None
    first_crossing = None

    # Align each frame to receptor coordinates in the reference frame, in memory.
    # Whole molecules and consistent imaging are expected from preprocessing;
    # merely setting --use-pbc does not unwrap broken molecules.
    aligner = align.AlignTraj(u, reference, select=args.align, in_memory=True)
    aligner.run()

    for ts in u.trajectory[::args.stride]:
        box = get_box(u, args.use_pbc)
        pcom = protein.center_of_mass()
        lcom = ligand.center_of_mass()
        if initial_ligand_com is None:
            initial_ligand_com = lcom.copy()

        com_distance = float(np.linalg.norm(lcom - pcom))
        com_displacement = float(np.linalg.norm(lcom - initial_ligand_com))
        ligand_rmsd = float(np.sqrt(np.mean(
            np.sum((ligand.positions-reference_ligand_coords)**2, axis=1)
        )))
        contacts = calculate_contact_count(protein, ligand, args.contact_cutoff, box)
        time_ps = float(ts.time) if np.isfinite(ts.time) else np.nan
        if first_crossing is None and com_distance >= args.escape_distance:
            first_crossing = (int(ts.frame), time_ps)

        records.append((int(ts.frame), time_ps, *lcom, com_distance,
                        com_displacement, ligand_rmsd, contacts))

    if not records:
        raise RuntimeError("No trajectory frames analyzed.")
    header = ("frame,time_ps,ligand_com_x,ligand_com_y,ligand_com_z,"
              "protein_ligand_com_distance_A,ligand_com_displacement_A,"
              "ligand_rmsd_aligned_A,contact_count")
    with open(args.output/"trajectory_observables.csv", "w",
              newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header.split(","))
        writer.writerows(records)

    data = np.asarray(records, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    t = data[:, 1] if np.all(np.isfinite(data[:, 1])) else data[:, 0]
    xlabel = "Time (ps)" if np.all(np.isfinite(data[:, 1])) else "Frame"
    for ax, y, label in zip(
        axes, (data[:, 5], data[:, 7], data[:, 8]),
        ("Protein–ligand COM distance (Å)", "Aligned ligand RMSD (Å)",
         f"Atom-pair contacts (<{args.contact_cutoff:g} Å)")
    ):
        ax.plot(t, y)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].axhline(args.escape_distance, color="black", ls="--",
                    label="First-crossing threshold")
    axes[0].legend()
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(args.output/"trajectory_observables.png", dpi=200)
    plt.close(fig)

    with open(args.output/"summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Analyzed frames: {len(records)}\n")
        fh.write(f"Protein selection: {args.protein}\n")
        fh.write(f"Ligand selection: {args.ligand}\n")
        fh.write(f"Alignment selection: {args.align}\n")
        fh.write(f"COM first-crossing threshold (A): {args.escape_distance}\n")
        fh.write(f"First crossing (frame, time_ps): {first_crossing}\n")
        fh.write("WARNING: First crossing is not necessarily irreversible escape.\n")
        fh.write("WARNING: PBC imaging, mass assignments, and alignment must be validated.\n")
    print((args.output/"summary.txt").read_text())


if __name__ == "__main__":
    main()
