"""Paper 9 — OpenMM 8 / PyTorch potentials (Eastman et al.).

Educational demonstration of a differentiable TorchScript potential using
OpenMM-Torch (openmmtorch.TorchForce). The model is NOT pretrained and is NOT
an accurate molecular force field. It serves to demonstrate interoperability.

Run:
  python 10_ml_extension/torchforce_demo.py --check-only
  python 10_ml_extension/torchforce_demo.py --run-openmm

The first mode needs only PyTorch and checks energies/forces by autograd.
The second requires OpenMM + OpenMM-Torch installed in a compatible environment.
Units in the OpenMM integration are nm and kJ/mol; 3 particles in vacuum.
"""
import argparse
from pathlib import Path
import torch


class ToyHarmonicPotential(torch.nn.Module):
    """Translationally invariant pairwise quadratic potential."""

    def __init__(self, stiffness: float = 100.0, equilibrium_nm: float = 0.15):
        super().__init__()
        self.register_buffer("k", torch.tensor(stiffness, dtype=torch.float64))
        self.register_buffer("r0", torch.tensor(equilibrium_nm, dtype=torch.float64))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        d01 = torch.linalg.vector_norm(positions[0] - positions[1])
        d12 = torch.linalg.vector_norm(positions[1] - positions[2])
        return 0.5 * self.k * ((d01 - self.r0) ** 2 + (d12 - self.r0) ** 2)


def test_autograd(model: torch.nn.Module):
    xyz = torch.tensor([[-0.15, 0., 0.], [0.01, 0., 0.], [0.15, 0., 0.]],
                       dtype=torch.float64, requires_grad=True)
    energy = model(xyz)
    forces = -torch.autograd.grad(energy, xyz)[0]
    assert torch.isfinite(energy) and torch.isfinite(forces).all()
    assert torch.linalg.vector_norm(forces.sum(dim=0)) < 1e-8
    assert torch.linalg.vector_norm(forces) > 0
    print(f"PyTorch energy: {energy.item():.9f} kJ/mol")
    print("PyTorch forces (kJ/mol/nm):\n", forces.detach().numpy())
    return xyz.detach()


def run_openmm(model_path: Path, positions: torch.Tensor):
    try:
        import openmm as mm
        from openmm import unit
        from openmmtorch import TorchForce
    except ImportError as e:
        raise RuntimeError("Need compatible OpenMM and OpenMM-Torch installations") from e

    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0 * unit.dalton)
    system.addForce(TorchForce(str(model_path)))
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions.numpy() * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    print("OpenMM energy:", state.getPotentialEnergy())
    print("OpenMM forces:", state.getForces())
    del context, integrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="PyTorch-only test (default)")
    parser.add_argument("--run-openmm", action="store_true", help="Also test OpenMM-Torch")
    parser.add_argument("--output", type=Path, default=Path("results_paper9"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = ToyHarmonicPotential().eval()
    sample_positions = test_autograd(model)
    scripted = torch.jit.script(model)
    model_path = args.output / "toy_potential.pt"
    scripted.save(str(model_path))
    print("Saved TorchScript model:", model_path)
    if args.run_openmm:
        run_openmm(model_path, sample_positions)
    else:
        print("OpenMM integration skipped; use --run-openmm when installed.")


if __name__ == "__main__":
    main()
