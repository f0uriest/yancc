"""Splitting DKE problems across devices by species and speed."""

import equinox as eqx
import jax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# mesh axis name for each distributed coordinate, keyed by its axorder letter
_MESH_AXES = {"s": "species", "x": "speed"}


def _validate_mesh(mesh: Mesh | None, ns: int, nx: int) -> None:
    """Check that a mesh can be used to split a problem with ns species, nx speeds."""
    if mesh is None:
        return
    if not isinstance(mesh, Mesh):
        raise TypeError(f"mesh must be a jax.sharding.Mesh, got {type(mesh)}")
    unknown = [name for name in mesh.axis_names if name not in _MESH_AXES.values()]
    if unknown:
        raise ValueError(
            f"mesh has unknown axis names {unknown}, expected a subset of "
            f"{list(_MESH_AXES.values())}"
        )
    if any(t != AxisType.Auto for t in mesh.axis_types):
        raise ValueError(
            "mesh axes must all have axis type jax.sharding.AxisType.Auto, eg "
            "jax.make_mesh(shape, names, axis_types=(AxisType.Auto,) * len(names))"
        )
    ds = mesh.shape.get("species", 1)
    dx = mesh.shape.get("speed", 1)
    if ns % ds:
        raise ValueError(
            f"species mesh axis of size {ds} doesn't divide the number of species {ns}"
        )
    if nx % dx:
        raise ValueError(
            f"speed mesh axis of size {dx} doesn't divide the number of speeds {nx}"
        )
    # Each device holds a contiguous piece of the distribution function, which is
    # ordered species first. The pieces only line up with blocks of whole species and
    # speeds if either speed isn't split, or every device has a single species.
    if dx > 1 and ds != ns:
        raise ValueError(
            "the speed mesh axis can only be split when the species mesh axis has one "
            f"device per species, got species axis size {ds} for {ns} species"
        )


def _constrain(x, mesh, spec):
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))


def _lead_axes(letters: str, mesh: Mesh) -> tuple[str, ...]:
    """Mesh axes for a leading dim flattened from the given coordinate letters.

    Only the leading run of distributed coordinates (species, speed) counts, since a
    flattened dim can only be split evenly along its outermost coordinates.
    """
    axes = []
    for letter in letters:
        if letter not in _MESH_AXES:
            break
        if _MESH_AXES[letter] in mesh.axis_names:
            axes.append(_MESH_AXES[letter])
    return tuple(axes)


def _shard_leading(x, mesh: Mesh, letters: str):
    """Split the leading dim of x, flattened from the given coordinates in order."""
    axes = _lead_axes(letters, mesh)
    return _constrain(x, mesh, P(axes or None, *([None] * (x.ndim - 1))))


def _shard_state(x, mesh: Mesh | None, axis: int = 0):
    """Split dim ``axis`` of x, a flattened (species, speed, ...) state."""
    if mesh is None:
        return x
    spec = [None] * x.ndim
    spec[axis] = _lead_axes("sx", mesh) or None
    return _constrain(x, mesh, P(*spec))


def _shard_sx(x, mesh: Mesh | None):
    """Split x, whose two leading dims are species and speed."""
    if mesh is None:
        return x
    names = [name if name in mesh.axis_names else None for name in _MESH_AXES.values()]
    return _constrain(x, mesh, P(*names, *([None] * (x.ndim - 2))))


def _is_array(x) -> bool:
    return eqx.is_array(x) and x.ndim > 0


def _replicate(tree, mesh: Mesh):
    """Keep every array of tree whole on each device of the mesh."""
    return jax.tree.map(lambda x: _constrain(x, mesh, P()) if _is_array(x) else x, tree)


def _shard_sx_leaves(tree, mesh: Mesh, ns: int, nx: int):
    """Split the arrays of tree whose two leading dims are (species, speed)."""
    return jax.tree.map(
        lambda x: _shard_sx(x, mesh) if _is_array(x) and x.shape[:2] == (ns, nx) else x,
        tree,
    )


def _has_shard(x) -> bool:
    return isinstance(x, eqx.Module) and hasattr(x, "_shard")


def _shard_dke(tree, mesh: Mesh | None):
    """Constrain the arrays of DKE operators and preconditioners to a mesh.

    Each module in ``tree`` with a ``_shard(mesh)`` method lays out its own arrays;
    arrays outside such modules are left for the compiler to place.
    """
    if mesh is None:
        return tree
    return jax.tree.map(
        lambda m: m._shard(mesh) if _has_shard(m) else m, tree, is_leaf=_has_shard
    )
