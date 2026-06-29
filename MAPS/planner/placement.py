"""Virtual-to-physical stage placement bindings."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.submesh import Submesh


@dataclass(frozen=True)
class StagePlacement:
    """Bind one rectangular virtual stage layout to physical mesh tiles."""

    stage_id: int
    virtual_submesh: object
    physical_submesh: Submesh
    virtual_to_physical: dict[int, int]

    def __post_init__(self) -> None:
        virtual_tile_ids = {tile.tile_id for tile in self.virtual_submesh.tiles}
        physical_tile_ids = set(self.physical_submesh.tile_ids)
        mapped_virtual_ids = set(self.virtual_to_physical)
        mapped_physical_ids = set(self.virtual_to_physical.values())

        if mapped_virtual_ids != virtual_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all virtual tiles"
            )
        if mapped_physical_ids != physical_tile_ids:
            raise ValueError(
                f"placement for stage {self.stage_id} does not cover all physical tiles"
            )

    def physical_tile_id(self, virtual_tile_id: int) -> int:
        """Return the physical tile id assigned to one virtual tile id."""

        return self.virtual_to_physical[virtual_tile_id]
