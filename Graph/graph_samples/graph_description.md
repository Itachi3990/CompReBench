## Graph Format Specification

DIRECTED/UNDIRECTED
<VERTEX_COUNT> <EDGE_COUNT>
<V1> <V2> <WEIGHT>
...

### Format Description

- **Line 1:** Graph type (`DIRECTED` or `UNDIRECTED`)
- **Line 2:** Vertex count and edge count separated by space
- **Line 3+:** One edge per line as `source destination weight` (or just `u v` if unweighted)