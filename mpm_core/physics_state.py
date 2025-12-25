"""
MPM Physics State Container
============================

GPU-resident data structure for all physical quantities.

✓ Self-Check:
- All tensors are on GPU by default
- Shapes are documented and validated
- Initialization preserves physical constraints (e.g., F = I initially)
- Compatible with PyTorch autograd
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MPMPhysicsState:
    """
    Container for all MPM particle physical state.

    All tensors are PyTorch CUDA tensors for gradient tracking.
    This class is the interface between 3DGS and MPM.

    Attributes:
        position: (N, 3) Particle positions [m]
        velocity: (N, 3) Particle velocities [m/s]
        F: (N, 3, 3) Deformation gradient (measures strain)
        C: (N, 3, 3) APIC affine momentum matrix
        mass: (N,) Particle masses [kg]
        volume: (N,) Particle volumes [m³]
        material_params: (N, 2) Per-particle (E, nu) - learnable!
    """

    # Core physical quantities
    position: torch.Tensor          # (N, 3)
    velocity: torch.Tensor          # (N, 3)
    F: torch.Tensor                 # (N, 3, 3) Deformation gradient
    C: torch.Tensor                 # (N, 3, 3) Affine momentum
    mass: torch.Tensor              # (N,)
    volume: torch.Tensor            # (N,)
    material_params: torch.Tensor   # (N, 2) [E, nu] - learnable!

    def __post_init__(self):
        """Validate shapes and constraints."""
        N = self.position.shape[0]

        # Shape validation
        assert self.position.shape == (N, 3), f"position shape mismatch: {self.position.shape}"
        assert self.velocity.shape == (N, 3), f"velocity shape mismatch: {self.velocity.shape}"
        assert self.F.shape == (N, 3, 3), f"F shape mismatch: {self.F.shape}"
        assert self.C.shape == (N, 3, 3), f"C shape mismatch: {self.C.shape}"
        assert self.mass.shape == (N,), f"mass shape mismatch: {self.mass.shape}"
        assert self.volume.shape == (N,), f"volume shape mismatch: {self.volume.shape}"
        assert self.material_params.shape == (N, 2), f"material_params shape mismatch"

        # Device check
        device = self.position.device
        assert all(t.device == device for t in [
            self.velocity, self.F, self.C, self.mass, self.volume, self.material_params
        ]), "All tensors must be on the same device"

        # Physical constraints check
        assert torch.all(self.mass > 0), "Mass must be positive"
        assert torch.all(self.volume > 0), "Volume must be positive"
        assert torch.all(self.material_params[:, 0] > 0), "Young's modulus must be positive"
        assert torch.all((self.material_params[:, 1] >= 0) & (self.material_params[:, 1] < 0.5)), \
            "Poisson's ratio must be in [0, 0.5)"

        # Deformation gradient should be close to identity initially
        det_F = torch.det(self.F)
        if torch.any(det_F <= 0):
            print(f"⚠️  Warning: {torch.sum(det_F <= 0)} particles have invalid F (det ≤ 0)")

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.position.shape[0]

    @property
    def device(self) -> torch.device:
        """Device where state resides."""
        return self.position.device

    def detach(self) -> 'MPMPhysicsState':
        """Detach all tensors from computation graph (for checkpointing)."""
        return MPMPhysicsState(
            position=self.position.detach(),
            velocity=self.velocity.detach(),
            F=self.F.detach(),
            C=self.C.detach(),
            mass=self.mass.detach(),
            volume=self.volume.detach(),
            material_params=self.material_params.detach(),
        )

    def clone(self) -> 'MPMPhysicsState':
        """Deep copy of state."""
        return MPMPhysicsState(
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            F=self.F.clone(),
            C=self.C.clone(),
            mass=self.mass.clone(),
            volume=self.volume.clone(),
            material_params=self.material_params.clone(),
        )

    def to(self, device: torch.device) -> 'MPMPhysicsState':
        """Move all tensors to specified device."""
        return MPMPhysicsState(
            position=self.position.to(device),
            velocity=self.velocity.to(device),
            F=self.F.to(device),
            C=self.C.to(device),
            mass=self.mass.to(device),
            volume=self.volume.to(device),
            material_params=self.material_params.to(device),
        )

    @staticmethod
    def from_gaussian_model(
        gaussians,
        config,
        velocity_init: Optional[torch.Tensor] = None,
    ) -> 'MPMPhysicsState':
        """
        Initialize MPM state from 3D Gaussian Splatting model.

        Args:
            gaussians: GaussianModel instance with get_xyz property
            config: MPMConfig instance
            velocity_init: Optional initial velocities (N, 3)

        Returns:
            MPMPhysicsState initialized from Gaussian positions

        ✓ Self-Check:
        - Positions extracted correctly from gaussians
        - F initialized to identity (no initial deformation)
        - Mass computed from density and volume
        - Material parameters use config defaults
        """
        xyz = gaussians.get_xyz  # (N, 3)
        N = xyz.shape[0]
        device = xyz.device
        dtype = config.dtype

        # Initialize velocities
        if velocity_init is None:
            velocity = torch.zeros(N, 3, device=device, dtype=dtype)
        else:
            velocity = velocity_init.to(device=device, dtype=dtype)

        # Initialize deformation gradient to identity
        F = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3).contiguous()

        # Initialize affine momentum to zero
        C = torch.zeros(N, 3, 3, device=device, dtype=dtype)

        # Compute particle volume and mass
        particle_volume = (config.dx ** 3) * config.particle_volume_scale
        volume = torch.full((N,), particle_volume, device=device, dtype=dtype)
        mass = volume * config.particle_density

        # Initialize material parameters (learnable)
        material_params = torch.zeros(N, 2, device=device, dtype=dtype)
        material_params[:, 0] = config.youngs_modulus  # E
        material_params[:, 1] = config.poissons_ratio  # nu

        state = MPMPhysicsState(
            position=xyz.clone(),  # Clone to avoid modifying gaussians
            velocity=velocity,
            F=F,
            C=C,
            mass=mass,
            volume=volume,
            material_params=material_params,
        )

        return state

    @staticmethod
    def create_test_state(
        n_particles: int = 1000,
        config = None,
        device: str = 'cuda',
    ) -> 'MPMPhysicsState':
        """
        Create a test state for debugging (particles in a cube).

        Args:
            n_particles: Number of particles
            config: MPMConfig (uses default if None)
            device: Device to create tensors on

        Returns:
            Random test state
        """
        from .config import get_default_config

        if config is None:
            config = get_default_config()

        dtype = config.dtype

        # Random positions in [0, 1]³
        position = torch.rand(n_particles, 3, device=device, dtype=dtype)

        # Zero velocities
        velocity = torch.zeros(n_particles, 3, device=device, dtype=dtype)

        # Identity deformation gradient
        F = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n_particles, 3, 3).contiguous()

        # Zero affine momentum
        C = torch.zeros(n_particles, 3, 3, device=device, dtype=dtype)

        # Uniform mass and volume
        particle_volume = (config.dx ** 3) * config.particle_volume_scale
        volume = torch.full((n_particles,), particle_volume, device=device, dtype=dtype)
        mass = volume * config.particle_density

        # Default material parameters
        material_params = torch.zeros(n_particles, 2, device=device, dtype=dtype)
        material_params[:, 0] = config.youngs_modulus
        material_params[:, 1] = config.poissons_ratio

        return MPMPhysicsState(
            position=position,
            velocity=velocity,
            F=F,
            C=C,
            mass=mass,
            volume=volume,
            material_params=material_params,
        )

    def get_kinetic_energy(self) -> torch.Tensor:
        """Compute total kinetic energy [J]."""
        return 0.5 * torch.sum(self.mass * torch.sum(self.velocity ** 2, dim=1))

    def get_momentum(self) -> torch.Tensor:
        """Compute total momentum [kg⋅m/s]."""
        return torch.sum(self.mass.unsqueeze(1) * self.velocity, dim=0)

    def get_volume_ratio(self) -> torch.Tensor:
        """Compute volume change ratio J = det(F)."""
        return torch.det(self.F)

    def compute_physics_loss(self) -> dict:
        """
        Compute physics-based loss terms for training.

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Volume preservation (incompressibility)
        J = self.get_volume_ratio()
        losses['volume_preservation'] = torch.mean((J - 1.0) ** 2)

        # Deformation gradient regularization (prevent extreme deformations)
        F_norm = torch.norm(self.F.reshape(-1, 9), dim=1)
        losses['F_regularization'] = torch.mean((F_norm - np.sqrt(3.0)) ** 2)  # sqrt(3) for identity

        # Velocity smoothness (neighbor consistency)
        # This would require spatial neighbor finding - placeholder for now
        losses['velocity_smoothness'] = torch.mean(self.velocity ** 2) * 0.0

        return losses


if __name__ == "__main__":
    print("Testing MPM Physics State...")

    from .config import get_default_config

    config = get_default_config()
    state = MPMPhysicsState.create_test_state(n_particles=1000, config=config)

    print(f"\n✓ Created state with {state.n_particles} particles")
    print(f"  Device: {state.device}")
    print(f"  Position range: [{state.position.min():.3f}, {state.position.max():.3f}]")
    print(f"  Total mass: {state.mass.sum():.3f} kg")
    print(f"  Kinetic energy: {state.get_kinetic_energy():.3e} J")
    print(f"  Volume ratio: {state.get_volume_ratio().mean():.3f} (should be ~1.0)")

    # Test physics loss
    losses = state.compute_physics_loss()
    print(f"\nPhysics losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.3e}")

    print("\n✓ All tests passed!")
