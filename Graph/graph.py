import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


class Graph:
    """A graph representation with adjacency-matrix output and common graph metrics."""

    def __init__(self, vertices: Optional[Iterable[str]] = None,
                 edges: Optional[Iterable[Tuple[str, str, int]]] = None,
                 directed: bool = False):
        self._directed = bool(directed)
        self.directed = self._directed

        self._adjacency: Dict[str, Dict[str, int]] = {}
        self.adjacency: Dict[str, Dict[str, int]] = self._adjacency

        self._edge_list: List[Tuple[str, str, int]] = []
        self.edges: List[Tuple[str, str, int]] = self._edge_list

        self._vertices: List[str] = []
        self.vertices: List[str] = self._vertices

        if vertices is not None:
            for vertex in vertices:
                self.add_vertex(vertex)

        if edges is not None:
            for u, v, weight in edges:
                self.add_edge(u, v, weight)

    def __repr__(self):
        return (
            f"Graph(vertices={self.get_number_of_vertices()}, "
            f"edges={self.get_number_of_edges()}, "
            f"directed={self._directed})"
        )

    @staticmethod
    def _natural_sort_key(vertex: str):
        match = re.search(r'\d+', str(vertex))
        if match:
            prefix = str(vertex)[:match.start()]
            number = int(match.group())
            return (prefix, number)
        return (str(vertex), 0)

    def add_vertex(self, vertex: str) -> None:
        vertex = str(vertex)
        if vertex not in self._adjacency:
            self._adjacency[vertex] = {}
            self._vertices.append(vertex)
            self.vertices = self._vertices

    def add_edge(self, u: str, v: str, weight: int = 1) -> None:
        u = str(u)
        v = str(v)
        weight = int(weight)

        self.add_vertex(u)
        self.add_vertex(v)

        if self._directed:
            self._adjacency[u][v] = weight
            self._edge_list.append((u, v, weight))
            self.edges = self._edge_list
        else:
            self._adjacency[u][v] = weight
            self._adjacency[v][u] = weight
            self._edge_list.append((u, v, weight))
            self.edges = self._edge_list

    def get_adjacency_matrix(self) -> List[List[int]]:
        """Return the adjacency matrix in a natural vertex ordering."""
        ordered_vertices = sorted(self._adjacency.keys(), key=self._natural_sort_key)
        index_by_vertex = {vertex: i for i, vertex in enumerate(ordered_vertices)}
        matrix = [[0 for _ in ordered_vertices] for _ in ordered_vertices]

        for vertex in ordered_vertices:
            row_index = index_by_vertex[vertex]
            for neighbor, weight in self._adjacency[vertex].items():
                if neighbor not in index_by_vertex:
                    continue
                column_index = index_by_vertex[neighbor]
                matrix[row_index][column_index] = int(weight)

        return matrix

    def get_edge_list(self) -> List[Tuple[str, str, int]]:
        return list(self._edge_list)

    def is_directed(self) -> bool:
        return self._directed

    def get_number_of_vertices(self) -> int:
        return len(self._adjacency)

    def get_number_of_edges(self) -> int:
        if self._directed:
            return sum(len(neighbors) for neighbors in self._adjacency.values())

        seen_pairs = set()
        count = 0
        for u, neighbors in self._adjacency.items():
            for v in neighbors:
                if u == v:
                    count += 1
                    continue
                pair = tuple(sorted((u, v)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    count += 1
        return count

    def has_self_loops(self) -> bool:
        for vertex, neighbors in self._adjacency.items():
            if vertex in neighbors:
                return True
        return False

    def has_parallel_edges(self) -> bool:
        counts = {}
        for u, v, _ in self._edge_list:
            key = (u, v) if self._directed else tuple(sorted((u, v)))
            counts[key] = counts.get(key, 0) + 1
        return any(value > 1 for value in counts.values())

    def get_indegree_and_outdegree_of_every_vertex(self) -> Dict[str, Dict[str, int]]:
        indegree = {vertex: 0 for vertex in self._adjacency}
        outdegree = {vertex: 0 for vertex in self._adjacency}

        for vertex, neighbors in self._adjacency.items():
            outdegree[vertex] = len(neighbors)
            for neighbor in neighbors:
                indegree[neighbor] = indegree.get(neighbor, 0) + 1

        summary = {}
        for vertex in self._adjacency:
            if self._directed:
                summary[vertex] = {
                    "in_degree": indegree.get(vertex, 0),
                    "out_degree": outdegree.get(vertex, 0),
                }
            else:
                degree = outdegree.get(vertex, 0)
                summary[vertex] = {
                    "in_degree": degree,
                    "out_degree": degree,
                    "degree": degree,
                }
        return summary

    def get_degree_summary(self) -> Dict[str, Dict[str, int]]:
        return self.get_indegree_and_outdegree_of_every_vertex()

    @classmethod
    def from_adjacency_dict(cls, graph_dict: Dict[str, Dict[str, int]], directed: Optional[bool] = None):
        graph = cls(directed=bool(directed))
        for u, neighbors in graph_dict.items():
            graph.add_vertex(u)
            for v, weight in neighbors.items():
                graph.add_vertex(v)
                graph.add_edge(u, v, int(weight))
        return graph

    @classmethod
    def from_file(cls, file_path: str, directed: Optional[bool] = None):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            return cls(directed=bool(directed))

        if lines[0] in ("DIRECTED", "UNDIRECTED"):
            is_directed = lines[0] == "DIRECTED"
            edge_lines = lines[2:]
        else:
            is_directed = bool(directed)
            edge_lines = lines

        graph = cls(directed=is_directed)
        for edge_line in edge_lines:
            tokens = edge_line.split()
            if len(tokens) < 2:
                continue
            u, v = tokens[0], tokens[1]
            weight = int(tokens[2]) if len(tokens) >= 3 else 1
            graph.add_edge(u, v, weight)
        return graph


def calculate_adjacency_matrix(graph, directed=None):
    """Backward-compatible matrix helper for the original script."""
    if isinstance(graph, Graph):
        return graph.get_adjacency_matrix()
    if isinstance(graph, dict):
        return Graph.from_adjacency_dict(graph, directed=bool(directed)).get_adjacency_matrix()
    raise TypeError("graph must be a Graph instance or a dict representation.")


def read_graph_from_file(file_path, directed=None):
    """Backward-compatible loader returning a Graph object."""
    return Graph.from_file(file_path, directed=directed)


def process_dataset(base_dir, directed):
    erroneous_graphs = []
    graph_type = "directed" if directed else "undirected"

    print(f"\n========== Processing {graph_type} graphs ==========")

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        graph_file = os.path.join(folder_path, f"graph_{folder}.txt")
        print(f"Processing {graph_file}")

        try:
            graph = read_graph_from_file(graph_file, directed)
            adjacency_matrix = graph.get_adjacency_matrix()
            output_file = os.path.join(folder_path, f"adjacency_matrix_{folder}.txt")

            with open(output_file, "w", encoding="utf-8") as file:
                file.write(str(adjacency_matrix))

            print(f"Saved {output_file}")
        except Exception as exc:
            erroneous_graphs.append(graph_file)
            print(f"ERROR: {exc}")

        print()

    if erroneous_graphs:
        print(f"{graph_type.upper()} ERRORS:")
        for graph_path in erroneous_graphs:
            print(" -", graph_path)
    else:
        print(f"All {graph_type} graphs processed successfully! NO ERRORS!")


def delete_adjacency_matrices(base_dir):
    """Delete all adjacency matrix files under a dataset directory."""
    if not os.path.exists(base_dir):
        return

    deleted = 0
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        matrix_file = os.path.join(folder_path, f"adjacency_matrix_{folder}.txt")
        if os.path.exists(matrix_file):
            os.remove(matrix_file)
            print(f"Deleted {matrix_file}")
            deleted += 1

    print(f"Deleted {deleted} adjacency matrix file(s) from {base_dir}.")


def delete_all_adjacency_matrices():
    project_root = os.path.dirname(os.path.abspath(__file__))
    delete_adjacency_matrices(os.path.join(project_root, "graph_samples", "undirected"))
    delete_adjacency_matrices(os.path.join(project_root, "graph_samples", "directed"))


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    process_dataset(
        os.path.join(project_root, "graph_samples", "undirected"),
        directed=False,
    )

    process_dataset(
        os.path.join(project_root, "graph_samples", "directed"),
        directed=True,
    )