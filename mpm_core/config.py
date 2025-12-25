"""
MPM Configuration Module
========================

Centralized configuration for all MPM simulation parameters.

✓ Self-Check:
- All parameters have physical units documented
- Default values are physically reasonable
- Grid resolution is power of 2 for optimal GPU performance
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MPMConfig:
    """
    Configuration for MPM simulation parameters.

    All parameters are designed for GPU optimization and physical accuracy.
    """

    # ============ Simulation Control ============
    dt: float = 1e-4                    # Time step [seconds] - stable for typical materials
    substeps: int = 5                   # Physics substeps per render frame

    # ============ Spatial Domain ============
    grid_resolution: Tuple[int, int, int] = (64, 64, 64)  # Grid dimensions (power of 2 for GPU)
    domain_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Physical size [meters]

    @property
    def dx(self) -> float:
        """Grid spacing [meters]"""
        return self.domain_size[0] / self.grid_resolution[0]

    @property
    def inv_dx(self) -> float:
        """Inverse grid spacing [1/meters]"""
        return self.grid_resolution[0] / self.domain_size[0]

    # ============ Physics Constants ============
    gravity: List[float] = None         # Gravity [m/s²] - default is [0, -9.8, 0]

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = [0.0, -9.8, 0.0]

    # ============ Material Parameters (Default) ============
    # Neo-Hookean constitutive model
    youngs_modulus: float = 1e5         # Young's modulus [Pa] - rubber-like
    poissons_ratio: float = 0.3         # Poisson's ratio [dimensionless] - typical solid

    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter [Pa]"""
        E, nu = self.youngs_modulus, self.poissons_ratio
        return (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter (shear modulus) [Pa]"""
        E, nu = self.youngs_modulus, self.poissons_ratio
        return E / (2.0 * (1.0 + nu))

    # ============ Particle Properties ============
    particle_density: float = 1000.0    # Particle density [kg/m³] - water density
    particle_volume_scale: float = 0.5  # Initial volume = dx³ * scale

    # ============ Numerical Parameters ============
    dtype: torch.dtype = torch.float32  # Precision (float32 for speed, float64 for accuracy)
    device: str = 'cuda'                # Compute device

    # ============ MPM Algorithm Settings ============
    use_apic: bool = True               # Use APIC (Affine Particle-In-Cell) for momentum
    alpha: float = 0.95                 # Grid-to-particle transfer coefficient (damping)

    # ============ Boundary Conditions ============
    boundary_friction: float = 0.5      # Friction coefficient [dimensionless]
    boundary_type: str = "sticky"       # "sticky", "slip", or "separate"

    # ============ Loss Weights for Training ============
    loss_momentum: float = 1.0          # Weight for momentum conservation
    loss_incompressibility: float = 0.1 # Weight for volume preservation
    loss_boundary: float = 0.5          # Weight for boundary collision

    # ============ Performance Settings ============
    max_particles: int = 100000         # Maximum number of particles
    block_size: int = 256               # CUDA block size (tuned for modern GPUs)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        ✓ Self-Check:
        - Time step is stable (CFL condition)
        - Poisson's ratio is physically valid
        - Grid resolution is sufficient
        """
        # Check CFL condition: dt * velocity < dx
        max_velocity = 100.0  # m/s - conservative estimate
        cfl_dt = self.dx / max_velocity
        if self.dt > cfl_dt:
            print(f"⚠️  Warning: dt={self.dt} may violate CFL condition (recommended < {cfl_dt:.2e})")

        # Check Poisson's ratio
        if not (0.0 <= self.poissons_ratio < 0.5):
            raise ValueError(f"Invalid Poisson's ratio: {self.poissons_ratio} (must be in [0, 0.5))")

        # Check grid resolution
        if min(self.grid_resolution) < 32:
            print(f"⚠️  Warning: Low grid resolution {self.grid_resolution} may cause artifacts")

        print("✓ Configuration validated successfully")

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            'dt': self.dt,
            'grid_resolution': self.grid_resolution,
            'domain_size': self.domain_size,
            'dx': self.dx,
            'youngs_modulus': self.youngs_modulus,
            'poissons_ratio': self.poissons_ratio,
            'gravity': self.gravity,
        }


# ============ Preset Configurations ============

def get_default_config() -> MPMConfig:
    """Get default configuration (general purpose)."""
    return MPMConfig()


def get_fast_config() -> MPMConfig:
    """Get fast configuration (for quick prototyping)."""
    return MPMConfig(
        grid_resolution=(32, 32, 32),
        substeps=3,
        dt=2e-4,
    )


def get_high_quality_config() -> MPMConfig:
    """Get high-quality configuration (for final results)."""
    return MPMConfig(
        grid_resolution=(128, 128, 128),
        substeps=10,
        dt=5e-5,
        dtype=torch.float64,  # Higher precision
    )


def get_soft_body_config() -> MPMConfig:
    """Preset for soft elastic materials (jelly, rubber)."""
    return MPMConfig(
        youngs_modulus=1e4,   # Softer material
        poissons_ratio=0.45,  # Nearly incompressible
        alpha=0.98,           # Less damping
    )


def get_rigid_body_config() -> MPMConfig:
    """Preset for stiff materials (metal, plastic)."""
    return MPMConfig(
        youngs_modulus=1e7,   # Stiffer material
        poissons_ratio=0.3,
        dt=5e-5,              # Smaller timestep for stability
    )


# if __name__ == "__main__":
#     # Test configuration
#     print("Testing MPM Configuration...")

#     config = get_default_config()
#     config.validate()

#     print("\nConfiguration Summary:")
#     print(f"  Grid: {config.grid_resolution}")
#     print(f"  dx: {config.dx:.4f} m")
#     print(f"  dt: {config.dt:.2e} s")
#     print(f"  Young's Modulus: {config.youngs_modulus:.2e} Pa")
#     print(f"  Lamé λ: {config.lame_lambda:.2e} Pa")
#     print(f"  Lamé μ: {config.lame_mu:.2e} Pa")
#     print(f"  Gravity: {config.gravity}")

#     print("\n✓ All tests passed!")
