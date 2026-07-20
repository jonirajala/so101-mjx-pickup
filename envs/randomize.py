"""Dynamics domain randomization for SO101PickCube.

`domain_randomize(model, rng) -> (batched_model, in_axes)` in the brax/playground
convention: vmap over rng to produce per-env model fields, mark randomized leaves
with in_axes 0 (all else None), and Brax vmaps `step` over axis 0.

The dominant sim-to-real lever is actuator stiffness: kp is randomized (with kv
kept ~critically-damped) over a wide band from a soft servo up to the real Feetech
STS3215, plus joint damping/frictionloss/armature and the cube's size/mass/friction.
All DR is *multiplicative around the validated nominal* rather than absolute, so
the grasp-validated contact regime stays in-distribution.

Arm dofs/actuators are indices 0:6 (SO-101: shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper); box free-joint dofs are 6:12. Box body/geom ids
must be passed by name by the caller (the module constants below are fallbacks only).
"""
import jax
import jax.numpy as jp
from mujoco import mjx

_ARM = slice(0, 6)        # arm + gripper actuators / dofs (positional from 0 — stable)
_ARM_BODIES = slice(1, 7)  # base..gripper link bodies (positional from 1 — stable)
# Default box ids for the current model; callers should pass ids resolved by name
# (`mj.body("box").id` / `mj.geom("box").id`) so a future geom/body insertion can't
# silently mis-target the cube DR. These are fallbacks only.
_BOX_BODY = 8
_BOX_GEOM = 32


def domain_randomize(model: mjx.Model, rng: jax.Array, box_body=_BOX_BODY, box_geom=_BOX_GEOM):
    @jax.vmap
    def rand(rng):
        # --- Actuator stiffness: effective kp ~ LogU(50, 1000) ---
        # The model nominal kp is 998.22 (sts3215 class in so101_mjx.xml), so the multiplier is
        # LogU(0.05, 1.0): a soft servo (kp~50) up to the validated real one (kp~998). position
        # actuator: gain=[kp,0,0], bias=[0,-kp,-kv]; keep kv ~critically damped via
        # kv *= sqrt(kp_mult) (kv ∝ sqrt(kp) at fixed inertia/dampratio).
        # GRIPPER (index 5) kp multiplier is floored at 0.3 (kp~300): a too-soft jaw physically
        # CANNOT hold the cube, so training on it teaches grips that slip.
        rng, k = jax.random.split(rng)
        kp_lo = jp.full((6,), jp.log(0.05)).at[5].set(jp.log(0.3))
        kp_mult = jp.exp(jax.random.uniform(k, (6,), minval=kp_lo, maxval=jp.log(1.0)))
        gainprm = model.actuator_gainprm.at[_ARM, 0].set(
            model.actuator_gainprm[_ARM, 0] * kp_mult)
        biasprm = model.actuator_biasprm.at[_ARM, 1].set(
            model.actuator_biasprm[_ARM, 1] * kp_mult)              # -kp
        biasprm = biasprm.at[_ARM, 2].set(
            model.actuator_biasprm[_ARM, 2] * jp.sqrt(kp_mult))     # -kv

        # --- Joint damping: arm dofs get U(0.0, 0.3) (nominal is 0) ---
        rng, k = jax.random.split(rng)
        dof_damping = model.dof_damping.at[_ARM].set(
            jax.random.uniform(k, (6,), minval=0.0, maxval=0.3))

        # --- Joint dry friction: *= U(0.5, 2.0) (nominal 0.1) ---
        rng, k = jax.random.split(rng)
        dof_frictionloss = model.dof_frictionloss.at[_ARM].set(
            model.dof_frictionloss[_ARM]
            * jax.random.uniform(k, (6,), minval=0.5, maxval=2.0))

        # --- Reflected inertia / armature: *= U(0.8, 1.5) (nominal 0.1) ---
        rng, k = jax.random.split(rng)
        dof_armature = model.dof_armature.at[_ARM].set(
            model.dof_armature[_ARM]
            * jax.random.uniform(k, (6,), minval=0.8, maxval=1.5))

        # --- Cube tangential friction: nominal × U(0.6, 1.4). Keep the band multiplicative
        # around the nominal: friction is invisible to the camera, and a much more slippery
        # band makes the jaw squirt the cube away on close, collapsing grasp learning. ---
        rng, k = jax.random.split(rng)
        geom_friction = model.geom_friction.at[box_geom, 0].set(
            model.geom_friction[box_geom, 0]
            * jax.random.uniform(k, minval=0.6, maxval=1.4))

        # --- Box SIZE: anisotropic flat-cuboid half-extents, hx U(0.022,0.030),
        # hy U(0.016,0.024), hz U(0.008,0.013) m (= 4.4-6.0 x 3.2-4.8 x 1.6-2.6 cm).
        # Spawn height tracks geom_size[box,2] in the env reset, so taller/shorter
        # boxes still rest on the floor. ---
        rng, k = jax.random.split(rng)
        new_half = jax.random.uniform(k, (3,), minval=jp.array([0.022, 0.016, 0.008]),
                                      maxval=jp.array([0.030, 0.024, 0.013]))
        geom_size = model.geom_size.at[box_geom, :3].set(new_half)

        # --- Cube mass: CONSTANT DENSITY 200 kg/m^3 — mass = density * volume of the
        # DR'd cuboid (8*hx*hy*hz), so density stays fixed as size varies. ---
        body_mass = model.body_mass.at[box_body].set(
            200.0 * 8.0 * new_half[0] * new_half[1] * new_half[2])
        # --- Arm link masses: *= U(0.9, 1.1) ---
        rng, k = jax.random.split(rng)
        body_mass = body_mass.at[_ARM_BODIES].set(
            body_mass[_ARM_BODIES]
            * jax.random.uniform(k, (6,), minval=0.9, maxval=1.1))

        return (gainprm, biasprm, dof_damping, dof_frictionloss,
                dof_armature, geom_friction, body_mass, geom_size)

    (gainprm, biasprm, dof_damping, dof_frictionloss,
     dof_armature, geom_friction, body_mass, geom_size) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "dof_damping": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "geom_friction": 0,
        "body_mass": 0,
        "geom_size": 0,
    })
    model = model.tree_replace({
        "actuator_gainprm": gainprm,
        "actuator_biasprm": biasprm,
        "dof_damping": dof_damping,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "geom_size": geom_size,
    })
    return model, in_axes
