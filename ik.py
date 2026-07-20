"""Damped least-squares position IK for the SO-101 gripper site (CPU/numpy + MuJoCo).

5-DOF arm: we solve for gripper-site *position* only (orientation is left to the arm).
Used to: pick a sane home pose, define the reachable cube-spawn region, and pose the
arm around a resting cube for the scripted grasp test. Not used at RL time.
"""
import numpy as np
import mujoco

# SO-101 (so101_mjx.xml) arm joints, base->wrist; the jaw joint is "gripper".
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Multi-restart seeds for the home/spawn-region IK (no q_init given). The 5-DOF arm has
# folded-elbow IK branches that trap a single seed in a prongs-up local minimum; trying a
# spread and keeping the best solution reaches the workspace reliably.
_SEEDS = (
    np.array([0.0, -1.0, 1.0, 1.0, -1.57]),
    np.array([1.79, -0.97, 0.68, 1.16, -0.16]),
    np.array([0.0, -1.0, 1.3, 1.3, 0.0]),
    np.array([0.0, -0.6, 0.8, 1.4, 0.0]),
    np.array([0.0, -1.4, 1.5, 1.0, 1.5]),
)


def _ik_from(m, d, target_pos, arm_qadr, arm_dofs, lo, hi, gid, q0, iters, tol, damping):
    d.qpos[arm_qadr] = q0
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target_pos - d.site_xpos[gid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp, None, gid)
        J = jacp[:, arm_dofs]                       # 3 x 5
        # damped least squares: dq = Jt (J Jt + l^2 I)^-1 err
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        d.qpos[arm_qadr] = np.clip(d.qpos[arm_qadr] + J.T @ np.linalg.solve(JJt, err), lo, hi)
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(target_pos - d.site_xpos[gid]))


def solve_ik(m, target_pos, q_init=None, jaw=0.4, iters=200, tol=1e-4, damping=0.05):
    """Return a full qpos (arm + jaw + box) placing the gripper site at target_pos.

    q_init given -> single solve from it (used to refine a lift pose). q_init None -> multi-seed
    restart, returning the lowest-residual solution.
    """
    d = mujoco.MjData(m)
    gid = m.site("gripper").id
    arm_dofs = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
    arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
    jaw_qadr = m.jnt_qposadr[m.joint("gripper").id]
    arm_ids = [m.joint(j).id for j in ARM_JOINTS]
    lo, hi = m.jnt_range[arm_ids, 0], m.jnt_range[arm_ids, 1]
    d.qpos[jaw_qadr] = jaw

    if q_init is not None:
        d.qpos[:] = q_init
        d.qpos[jaw_qadr] = jaw
        err = _ik_from(m, d, target_pos, arm_qadr, arm_dofs, lo, hi, gid,
                       np.array(q_init)[arm_qadr], iters, tol, damping)
        return np.array(d.qpos), err

    best_q, best_err = None, np.inf
    for seed in _SEEDS:
        err = _ik_from(m, d, target_pos, arm_qadr, arm_dofs, lo, hi, gid,
                       seed, iters, tol, damping)
        if err < best_err:
            best_err, best_q = err, d.qpos.copy()
        if best_err < tol:
            break
    return np.array(best_q), best_err
