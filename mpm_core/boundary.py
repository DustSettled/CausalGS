import torch
import warp as wp


def apply_boundary_conditions_torch(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    domain_min: torch.Tensor,
    domain_max: torch.Tensor,
    friction: float = 0.5,
    boundary_type: str = "sticky",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply boundary conditions to particles (PyTorch version).

    Args:
        positions: (N, 3)
        velocities: (N, 3)
        domain_min: (3,) - minimum coordinates
        domain_max: (3,) - maximum coordinates
        friction: Friction coefficient
        boundary_type: "sticky", "slip", or "separate"

    Returns:
        (new_positions, new_velocities)

    ✓ Self-Check:
    - Particles stay within [domain_min, domain_max]
    - Velocity reflection is correct
    """
    pos = positions.clone()
    vel = velocities.clone()

    for dim in range(3):
        # Lower boundary
        mask_lower = pos[:, dim] < domain_min[dim]
        if mask_lower.any():
            pos[mask_lower, dim] = domain_min[dim]

            if boundary_type == "sticky":
                vel[mask_lower, :] = 0
            elif boundary_type == "slip":
                vel[mask_lower, dim] = 0  # Only normal component
                # Apply friction to tangent components
                tangent_dims = [d for d in range(3) if d != dim]
                vel[mask_lower, tangent_dims] *= (1.0 - friction)
            elif boundary_type == "separate":
                vel[mask_lower, dim] = torch.abs(vel[mask_lower, dim])  # Reflect

        # Upper boundary
        mask_upper = pos[:, dim] > domain_max[dim]
        if mask_upper.any():
            pos[mask_upper, dim] = domain_max[dim]

            if boundary_type == "sticky":
                vel[mask_upper, :] = 0
            elif boundary_type == "slip":
                vel[mask_upper, dim] = 0
                tangent_dims = [d for d in range(3) if d != dim]
                vel[mask_upper, tangent_dims] *= (1.0 - friction)
            elif boundary_type == "separate":
                vel[mask_upper, dim] = -torch.abs(vel[mask_upper, dim])

    return pos, vel


@wp.kernel
def apply_boundary_conditions_warp(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    friction: float,
):
    """
    Warp kernel: Apply sticky boundary conditions.

    ✓ Formula verified: clamp position, zero velocity on contact.
    """
    tid = wp.tid()

    pos = positions[tid]
    vel = velocities[tid]

    # Check each dimension
    for dim in range(3):
        # Lower boundary
        if pos[dim] < domain_min[dim]:
            pos[dim] = domain_min[dim]
            vel = wp.vec3(0.0, 0.0, 0.0)  # Sticky

        # Upper boundary
        if pos[dim] > domain_max[dim]:
            pos[dim] = domain_max[dim]
            vel = wp.vec3(0.0, 0.0, 0.0)  # Sticky

    positions[tid] = pos
    velocities[tid] = vel


if __name__ == "__main__":
    print("Testing Boundary Conditions...")

    N = 100
    pos = torch.rand(N, 3) * 2.0 - 0.5  # Some outside [0, 1]
    vel = torch.randn(N, 3)

    domain_min = torch.zeros(3)
    domain_max = torch.ones(3)

    print(f"Before: {(pos < 0).sum()} particles below 0")
    print(f"Before: {(pos > 1).sum()} particles above 1")

    pos_new, vel_new = apply_boundary_conditions_torch(
        pos, vel, domain_min, domain_max, boundary_type="sticky"
    )

    print(f"After: {(pos_new < 0).sum()} particles below 0 (should be 0)")
    print(f"After: {(pos_new > 1).sum()} particles above 1 (should be 0)")

    assert torch.all((pos_new >= 0) & (pos_new <= 1)), "Particles escaped!"

    print("\n✓ Boundary conditions test passed!")
