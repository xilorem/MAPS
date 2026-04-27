The `MAPS.arch/` contains the hardware abstraction layer used by MAPS.

## Package contents

```text
MAPS/arch/
    device.py
    memory.py
    tile.py
    mesh.py
    noc.py
```

## Mesh
A `Mesh` represents a rectangular grid of tiles; this is the high level object used by the tool for the mapping phase.
 A set of pre-built meshes can be found in the `./MAPS/hw/chips` folder.

## Tile
A `Tile` is the atomic unit of the Mesh. The tiles have a personal id that follows row-major order inside the Mesh.

## Memory
`L1Memory` describes tile local-memory. In order for a Device to start computation, the input data must reside in the local memory. Tiles can freely read and write data from/to other Tile's local memory.

`L2Memory` describes global shared-memory. It's possible to define a set of L2 access points that represent the physical location where IO from the NoC to the global memory are placed.

## Device
A `Device` is the computational engine of the Tile. Multiple devices can be instantiated inside a Tile and the best will be selected for computation. In order for an operation to use a Device, that operation needs to be supported in the Device itself.

A set of pre-built devices can be found in the `./MAPS/hw/devices` folder.

## NoC
The `NoC` describes the topology used for data movement between Tiles, from tiles to local memory and viceversa. 

An `NoCNode` describes the physical location of hardware components with respect to the NoC. 

To attach pheripherals to the NoC an `NoCEndpoint` is used. This allows the peripheral to import or export data from/to the NoC.
The `MAPS.arch/` contains the hardware abstraction layer used by MAPS.

## Package contents

```text
MAPS/arch/
    device.py
    memory.py
    tile.py
    mesh.py
    noc.py
```

## Mesh
A `Mesh` represents a rectangular grid of tiles. A set of pre-built meshes can be found in the `./MAPS/hw/chips` folder.

## Tile
A `Tile` is the atomic unit of the Mesh. The tiles have a personal id that follows row-major order inside the Mesh.

## Memory
`L1Memory` describes tile local-memory. In order for a Device to start computation, the input data must reside in the local memory. Tiles can freely read and write data from/to other Tile's local memory.

`L2Memory` describes global shared-memory. It's possible to define a set of L2 access points that represent the physical location where IO from the NoC to the global memory are placed.

## Device
A `Device` is the computational engine of the Tile. multiple devices can be instantiated inside a Tile and the best will be selected for computation. In order for an operation to use a Device, that operation needs to be supported in the Device itself.

A set of pre-built devices can be found in the `./MAPS/hw/devices` folder.

## NoC
The `NoC` describes the topology used for data movement between Tiles, from tiles to local memory and viceversa. 

An `NoCNode` describes the physical location of hardware components with respect to the NoC. 

To attach pheripherals to the NoC an `NoCEndpoint` is used. This allows the peripheral to import or export data from/to the NoC.

Data moves between Nodes via an `NoCLink`. A link can comprehend multiple `NoCChannels` that can be dedicated to specific data transfers.

An `NoCRoute` is used to describe a route between two Nodes.
Data moves between Nodes via an `NoCLink`. A link can comprehend multiple `NoCChannels` that can be dedicated to specific data transfers.