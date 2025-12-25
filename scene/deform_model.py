import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, CodeField
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, quaternion_multiply
from utils.velocity_field_utils import VelocityWarpper, SegVel
import einops

# ============ MPM Integration ============
try:
    from mpm_core import MPMConfig, MPMSimulator, MPMPhysicsState
    MPM_AVAILABLE = True
except ImportError:
    print("⚠️  MPM module not found. Physics constraints disabled.")
    print("   To enable: pip install warp-lang")
    MPM_AVAILABLE = False

class DeformModel:
    def __init__(self, is_blender=True, is_6dof=False, max_time=0.7, vel_start_time=0.0, light=True,
                 physics_code=16, use_mpm=False, mpm_config=None, use_velocity_field=True):
        """
        Enhanced DeformModel with optional MPM physics constraints.

        Args:
            use_mpm: Enable MPM physics regularization
            mpm_config: Optional MPMConfig (uses default if None)
            use_velocity_field: Enable velocity field integration (default: True)
                               Set to False for ablation: DeformNetwork + MPM only
        """
        self.d_gate = False
        self.v_gate = False
        deform_code_dim = physics_code # 16
        self.code_field = CodeField(D=4, W=128, input_ch=3, output_ch=deform_code_dim, multires=8).cuda()

        if light:
            self.deform = DeformNetwork(D=6, W=128, input_ch=3, hyper_ch=deform_code_dim, multires=8,
                                        is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()
        else:
            self.deform = DeformNetwork(D=8, W=256, input_ch=3, hyper_ch=deform_code_dim, multires=8,
                                        is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()
        self.vel_net = SegVel(deform_code_dim=deform_code_dim, hidden_dim=128, layers=5).cuda()
        self.vel = VelocityWarpper(self.vel_net)
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.max_time = max_time
        self.vel_start_time = vel_start_time

        # ============ Velocity Field Control ============
        self.use_velocity_field = use_velocity_field
        if not self.use_velocity_field:
            print("⚠️  Velocity Field DISABLED (Ablation Mode)")
            print("   Using: DeformNetwork + MPM only")
            print("   Velocity networks initialized but not used in forward pass")

        # ============ MPM Physics Integration ============
        self.use_mpm = use_mpm and MPM_AVAILABLE
        self.mpm_simulator = None
        self.mpm_state = None

        if self.use_mpm:
            if mpm_config is None:
                from mpm_core.config import get_default_config
                mpm_config = get_default_config()
                mpm_config.dt = 1/60  # Match rendering framerate

            self.mpm_simulator = MPMSimulator(mpm_config)
            print(f"✓ MPM physics enabled (grid: {mpm_config.grid_resolution})")
        else:
            if use_mpm and not MPM_AVAILABLE:
                print("⚠️  MPM requested but not available")

    def step_no_rot(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        max_time = self.max_time
        min_time = max(self.vel_start_time, dt)

        # ============ Ablation Mode: Disable Velocity Field ============
        if not self.use_velocity_field:
            # Only use DeformNetwork (no velocity field integration)
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
            if self.v_gate:
                d_xyz = d_xyz * gate
                d_rotation = d_rotation * gate
                d_rotation[..., 1:] = d_rotation[..., 1:] + 1
                d_scale = d_scale * gate
            return d_xyz, d_rotation, d_scale

        # ============ Normal Mode: Use Velocity Field ============
        if time_emb[0, 0] >= min_time and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(self.vel_start_time, max_time)
            gate = self.deform.get_gate(deform_code)
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            # in training, we only consider to integrate once, so we need to modify dt here
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            dt = time_emb[0, 0] - max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    def step_full_rot(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        max_time = self.max_time

        # ============ Ablation Mode: Disable Velocity Field ============
        if not self.use_velocity_field:
            # Only use DeformNetwork (no velocity field integration)
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
            if self.v_gate:
                d_xyz = d_xyz * gate
                d_rotation = d_rotation * gate
                d_rotation[..., 1:] = d_rotation[..., 1:] + 1
                d_scale = d_scale * gate
            return d_xyz, d_rotation, d_scale

        # ============ Normal Mode: Use Velocity Field ============
        if time_emb[0, 0] >= dt and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(0, max_time)
            gate = self.deform.get_gate(deform_code)
            d_xyz_deform, d_rotation, d_scale =  self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    def step(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        max_time = self.max_time

        # ============ Ablation Mode: Disable Velocity Field ============
        if not self.use_velocity_field:
            # Only use DeformNetwork (no velocity field integration)
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
            if self.v_gate:
                d_xyz = d_xyz * gate
                d_rotation = d_rotation * gate
                d_rotation[..., 1:] = d_rotation[..., 1:] + 1
                d_scale = d_scale * gate
            return d_xyz, d_rotation, d_scale

        # ============ Normal Mode: Use Velocity Field ============
        if time_emb[0, 0] > max_time:
            gate = self.deform.get_gate(deform_code)
            deform_time = torch.ones_like(time_emb) * max_time
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            deform_seg = self.code_field.seg(deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            gate = self.deform.get_gate(deform_code)
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        if self.v_gate:
            d_xyz = d_xyz * gate
            d_rotation = d_rotation * gate
            d_rotation[..., 1:] = d_rotation[..., 1:] + 1
            d_scale = d_scale * gate
        return d_xyz, d_rotation, d_scale

    # def step(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
    #     d_xyz, d_rotation, d_scale = self.deform(deform_code, time_emb)
    #     return d_xyz, d_rotation, d_scale

    # def base_vel_step(self, xyz, time_emb, dt=1/60):
    #     d_xyz = self.vel.integrate_pos(xyz, torch.zeros_like(time_emb), time_emb, dt) - xyz.detach()
    #     return d_xyz

    # def base_vel_step(self, xyz, time_emb, dt=1 / 5):
    #     xyz, d_rot = self.vel.integrate_pos(xyz, torch.zeros_like(time_emb), time_emb, dt, rot=True)
    #     d_xyz = xyz - xyz.detach()
    #     return d_xyz, d_rot

    # def vel_loss(self, deform_code, begin=0., end=1., tmax=0.75, device='cuda'):
    #     t = torch.rand(deform_code.shape[0], 1, device=device) * (end - begin) + begin
    #     x = torch.rand(deform_code.shape[0], 3, device=device) * 2 - 1
    #     # acc = self.deform.get_acc(deform_code, t)
    #     # loss = torch.mean(torch.norm(acc, dim=-1))
    #     acc = self.acc(deform_code.detach(), x)
    #     vel = self.deform.get_local_vel(deform_code.detach(), x, t)
    #     hess = self.deform.get_local_hessian(deform_code.detach(), x.copy(), t.copy())
    #     dvdt = hess[..., -1, -1]
    #     jac_v = hess[..., -1, :-1]
    #     v_jacv = einops.einsum(vel, jac_v, '... xyz, ... uvw xyz -> ... uvw')
    #     lhs = dvdt + v_jacv
    #     # interp
    #     mask = (t < tmax).squeeze(-1)
    #     loss_interp = torch.mean(torch.norm(lhs[mask].detach() - acc[mask], dim=-1))
    #     # extrap
    #     loss_extrap = torch.mean(torch.norm(lhs[~mask] - acc[~mask].detach(), dim=-1))
    #     loss = loss_interp + loss_extrap
    #     return loss

    def train_setting(self, training_args):
        # ============ Conditionally include velocity field parameters ============
        if self.use_velocity_field:
            params_list = (list(self.deform.parameters()) +
                          list(self.vel.parameters()) +
                          list(self.code_field.parameters()))
            print(f"✓ Optimizer: DeformNet + VelocityNet + CodeField")
        else:
            params_list = (list(self.deform.parameters()) +
                          list(self.code_field.parameters()))
            print(f"✓ Optimizer: DeformNet + CodeField (VelocityNet excluded)")
            print(f"  Note: Velocity network will NOT be optimized")

        l = [
            {'params': params_list,
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
        torch.save(self.code_field.state_dict(), os.path.join(out_weights_path, 'code_field.pth'))
        torch.save(self.vel.state_dict(), os.path.join(out_weights_path, 'vel.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        deform_weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(deform_weights_path))
        code_field_weights_path = os.path.join(model_path, "deform/iteration_{}/code_field.pth".format(loaded_iter))
        self.code_field.load_state_dict(torch.load(code_field_weights_path))
        vel_weights_path = os.path.join(model_path, "deform/iteration_{}/vel.pth".format(loaded_iter))
        self.vel.load_state_dict(torch.load(vel_weights_path))
        # try:
        #     self.vel.load_state_dict(torch.load(vel_weights_path))
        # except:
        #     print("No velocity weights found")

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # ============ MPM Physics Loss Methods ============

    def init_mpm_state(self, gaussians):
        """
        Initialize MPM physics state from Gaussian model.

        Call this once after loading/creating gaussians.
        """
        if not self.use_mpm:
            return

        self.mpm_state = MPMPhysicsState.from_gaussian_model(
            gaussians=gaussians,
            config=self.mpm_simulator.config,
        )
        print(f"✓ MPM state initialized ({self.mpm_state.n_particles} particles)")

    def compute_mpm_physics_loss(self, xyz, velocity=None):
        """
        Compute physics-based loss using MPM simulator.

        Args:
            xyz: Current Gaussian positions (N, 3)
            velocity: Optional current velocities (N, 3)

        Returns:
            Dictionary of physics loss components
        """
        if not self.use_mpm or self.mpm_state is None:
            return {}

        # Update MPM state with current positions
        self.mpm_state.position = xyz.clone()
        if velocity is not None:
            self.mpm_state.velocity = velocity.clone()

        # Compute physics losses
        return self.mpm_simulator.compute_physics_loss(self.mpm_state)

    def get_mpm_regularization_loss(self, xyz, velocity=None, weight=0.1):
        """
        Get weighted MPM regularization loss for training.

        Args:
            xyz: Current positions
            velocity: Optional velocities
            weight: Overall weight for MPM loss

        Returns:
            Scalar loss tensor (differentiable)
        """
        if not self.use_mpm:
            return torch.tensor(0.0, device=xyz.device)

        losses = self.compute_mpm_physics_loss(xyz, velocity)

        total_loss = weight * (
            losses.get('volume_preservation', 0.0) +
            losses.get('F_regularization', 0.0) * 0.1 +
            losses.get('boundary_penetration', 0.0) * 0.5
        )

        return total_loss

