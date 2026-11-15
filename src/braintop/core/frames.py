"""Explicit rigid coordinate frames for *display-space* 3D layer positions.

This is an SE(3) coordinate-frame contract, NOT intrinsic transport between
Riemannian manifolds. Every node position is local to its layer's frame.
"""
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator


Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]
Vector3 = Tuple[float, float, float]
IDENTITY: Matrix3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


class RigidFrame3D(BaseModel):
    """Right-handed, local-to-world orthogonal frame: world = R @ local + t.

    Frames provide consistent ambient coordinates, not a canonical
    correspondence, metric, or connection between distinct manifolds.
    """

    rotation: Matrix3 = Field(default=IDENTITY)
    translation: Vector3 = Field(default=(0., 0., 0.))

    @model_validator(mode="after")
    def validate_rigid(self) -> "RigidFrame3D":
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
            raise ValueError("Frame rotation and translation must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0):
            raise ValueError("Frame rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0):
            raise ValueError("Frame rotation must have determinant +1")
        return self

    def to_world(self, local: Vector3) -> Vector3:
        """Convert layer-local 3D display coordinates into world coordinates."""
        world = np.asarray(self.rotation) @ np.asarray(local) + np.asarray(self.translation)
        return tuple(float(v) for v in world)

    def reparameterized(self, q: Matrix3, s: Vector3) -> "RigidFrame3D":
        """Passive coordinate change y=Qx+s; preserve world=Rx+t.

        Thus R'=R Q^T and t'=t-R Q^T s. This changes the coordinate
        description, not the physical placement of the represented nodes.
        """
        change = RigidFrame3D(rotation=q, translation=s)
        r = np.asarray(self.rotation)
        q_ = np.asarray(change.rotation)
        s_ = np.asarray(change.translation)
        new_r = r @ q_.T
        new_t = np.asarray(self.translation) - new_r @ s_
        return RigidFrame3D(
            rotation=tuple(tuple(float(v) for v in row) for row in new_r),
            translation=tuple(float(v) for v in new_t),
        )
