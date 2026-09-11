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

    def add_edge(self, u: str, v: str, weight = 1) -> None:
        u = str(u)
        v = str(v)
        try:
            f_val = float(weight)
            weight = int(f_val) if f_val.is_integer() else f_val
        except (ValueError, TypeError):
            weight = 1

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

    def get_adjacency_matrix(self) -> List[List]:
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
                matrix[row_index][column_index] = weight

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

    def is_dense(self) -> bool:
        """
        Return True if the graph is dense, otherwise False.

        Density is defined as:

            undirected:
                m / [n(n-1)/2]

            directed:
                m / [n(n-1)]

        where:
            n = number of vertices
            m = number of unique non-self-loop edges

        A graph is considered dense when its density is at least 50%.
        Graphs with fewer than 2 vertices are considered sparse because
        density is not meaningfully defined for them.

        Self-loops are ignored when calculating density because the maximum
        possible edge counts above assume that self-loops are not allowed.
        """

        n = len(self._vertices)

        # Density is not meaningful for 0 or 1 vertex.
        if n < 2:
            return False

        # Count unique non-self-loop edges directly from the adjacency map.
        # The inner dictionaries map:
        #     neighbor -> weight
        #
        # This means duplicate insertions of the same edge should already
        # collapse to one adjacency entry.
        if self._directed:
            m = sum(
                1
                for u, neighbors in self._adjacency.items()
                for v in neighbors
                if u != v
            )

            max_edges = n * (n - 1)

        else:
            # In a properly represented undirected graph, every edge appears
            # twice in the adjacency structure:
            #
            #     u -> v
            #     v -> u
            #
            # Count non-self-loop adjacency entries and divide by 2.
            m = sum(
                1
                for u, neighbors in self._adjacency.items()
                for v in neighbors
                if u != v
            ) // 2

            max_edges = n * (n - 1) // 2

        density = m / max_edges

        # Standard, simple interpretation:
        # at least half of all possible edges => dense.
        return density >= 0.50

    def adjacency_matrix_matches(self, other_matrix: List[List[int]]) -> bool:
        expected = self.get_adjacency_matrix()
        if expected is None or other_matrix is None:
            return False
        if len(expected) != len(other_matrix):
            return False
        for i in range(len(expected)):
            if len(expected[i]) != len(other_matrix[i]):
                return False
            for j in range(len(expected[i])):
                val1 = expected[i][j]
                val2 = other_matrix[i][j]
                if isinstance(val1, float) and isinstance(val2, float):
                    if str(val1) == 'inf' and str(val2) == 'inf':
                        continue
                if val1 != val2:
                    return False
        return True

    def get_adjacency_matrix_entry_score(self, other_matrix: List[List[int]]) -> float:
        expected = self.get_adjacency_matrix()
        if expected is None or other_matrix is None:
            return 0.0
        if len(expected) != len(other_matrix):
            return 0.0
        total_entries = 0
        matching_entries = 0
        for i in range(len(expected)):
            if len(expected[i]) != len(other_matrix[i]):
                return 0.0
            for j in range(len(expected[i])):
                total_entries += 1
                val1 = expected[i][j]
                val2 = other_matrix[i][j]
                if isinstance(val1, float) and isinstance(val2, float):
                    if str(val1) == 'inf' and str(val2) == 'inf':
                        matching_entries += 1
                        continue
                if val1 == val2:
                    matching_entries += 1
        if total_entries == 0:
            return 1.0
        return matching_entries / total_entries

    def get_adjacency_matrix_row_score(self, other_matrix: List[List[int]]) -> float:
        expected = self.get_adjacency_matrix()
        if expected is None or other_matrix is None:
            return 0.0
        if len(expected) != len(other_matrix):
            return 0.0
        total_rows = len(expected)
        if total_rows == 0:
            return 1.0
        matching_rows = 0
        for i in range(len(expected)):
            if len(expected[i]) != len(other_matrix[i]):
                return 0.0
            row_match = True
            for j in range(len(expected[i])):
                val1 = expected[i][j]
                val2 = other_matrix[i][j]
                if isinstance(val1, float) and isinstance(val2, float):
                    if str(val1) == 'inf' and str(val2) == 'inf':
                        continue
                if val1 != val2:
                    row_match = False
                    break
            if row_match:
                matching_rows += 1
        return matching_rows / total_rows

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

    def get_edge_list_score(self, predicted_edges: List[Tuple]) -> float:
        """
        Score the predicted edge list against the ground truth with partial credit (0.0 – 1.0).

        Scoring logic (per-edge, then averaged):
          - Each ground-truth edge that appears correctly in the prediction gets full credit (1.0).
          - A ground-truth edge whose endpoint pair matches but whose weight is wrong gets half
            credit (0.5).
          - A ground-truth edge that is entirely missing gets no credit (0.0).
          - Extra (hallucinated) edges in the prediction are penalised by subtracting 1 credit per spurious edge from the total earned.  The final score is clamped to [0, 1].

        All comparisons normalise edge representations:
          - Strings are stripped and lower-cased for vertex names.
          - Weights are compared as rounded integers (int(round(w))).
          - For undirected graphs the edge (u, v) and (v, u) are treated as identical.
        """
        if predicted_edges is None:
            return 0.0

        def _normalise_edge(e):
            u = str(e[0]).strip().lower()
            v = str(e[1]).strip().lower()
            try:
                f_val = float(e[2])
                w = int(f_val) if f_val.is_integer() else round(f_val, 4)
            except Exception:
                w = 0
            if not self._directed and u > v:
                u, v = v, u
            return (u, v, w)

        def _endpoint_key(e):
            u, v, _ = _normalise_edge(e)
            return (u, v)

        gt_edges = []
        for edge in self.get_edge_list():
            gt_edges.append(_normalise_edge(edge))

        pred_edges = []
        for edge in predicted_edges:
            try:
                pred_edges.append(_normalise_edge(edge))
            except Exception:
                pass

        if not gt_edges:
            # No edges to score; penalise hallucinations
            return max(0.0, 1.0 - len(pred_edges))

        pred_set_full  = {}   # (u,v,w) -> count
        pred_set_pair  = {}   # (u,v)   -> list of weights
        for e in pred_edges:
            pred_set_full[e] = pred_set_full.get(e, 0) + 1
            key = (e[0], e[1])
            pred_set_pair.setdefault(key, []).append(e[2])

        earned = 0.0
        for gt_edge in gt_edges:
            pair_key = (gt_edge[0], gt_edge[1])
            if pred_set_full.get(gt_edge, 0) > 0:
                earned += 1.0
                pred_set_full[gt_edge] -= 1
            elif pair_key in pred_set_pair and pred_set_pair[pair_key]:
                # Endpoint match, wrong weight → half credit
                earned += 0.5
                pred_set_pair[pair_key].pop(0)
            # else: completely missing → 0 credit

        # Penalise spurious edges
        gt_set_full = {}
        for e in gt_edges:
            gt_set_full[e] = gt_set_full.get(e, 0) + 1

        spurious = 0
        for pred_edge in pred_edges:
            if gt_set_full.get(pred_edge, 0) > 0:
                gt_set_full[pred_edge] -= 1
            else:
                spurious += 1

        total_gt = len(gt_edges)
        raw_score = (earned - spurious) / total_gt
        return max(0.0, min(1.0, raw_score))

    def get_vertex_degrees_score(self, predicted_degrees: Dict) -> float:
        """
        Score the predicted vertex degree dictionary against ground truth with partial
        credit (0.0 – 1.0).

        Scoring logic:
          - For each ground-truth vertex:
              * Both in_degree and out_degree correct → 1.0 vertex credit.
              * Only one of them correct               → 0.5 vertex credit.
              * Both wrong or vertex missing           → 0.0 vertex credit.
          - Extra (hallucinated) vertices in the prediction are penalised by subtracting
            0.5 credit per spurious vertex.
          - Final score is (sum of vertex credits) / (number of ground-truth vertices),
            clamped to [0, 1].
        """
        if predicted_degrees is None:
            return 0.0

        expected_degrees = self.get_indegree_and_outdegree_of_every_vertex()
        if not expected_degrees:
            return 1.0 if not predicted_degrees else 0.0

        def _int(v):
            try:
                return int(round(float(v)))
            except Exception:
                return None

        earned = 0.0
        for vertex, exp_vals in expected_degrees.items():
            exp_in  = _int(exp_vals.get("in_degree",  0))
            exp_out = _int(exp_vals.get("out_degree", 0))

            pred_vals = predicted_degrees.get(vertex, None)
            if pred_vals is None:
                continue  # missing vertex → 0 credit

            pred_in  = _int(pred_vals.get("in_degree",  None))
            pred_out = _int(pred_vals.get("out_degree", None))

            correct = (pred_in == exp_in) + (pred_out == exp_out)
            earned += correct * 0.5   # 0, 0.5, or 1.0

        # Penalise hallucinated vertices
        spurious = sum(
            1 for v in predicted_degrees
            if v not in expected_degrees
        )
        raw = (earned - 0.5 * spurious) / len(expected_degrees)
        return max(0.0, min(1.0, raw))

    @classmethod
    def from_adjacency_dict(cls, graph_dict: Dict[str, Dict[str, int]], directed: Optional[bool] = None):
        graph = cls(directed=bool(directed))
        for u, neighbors in graph_dict.items():
            graph.add_vertex(u)
            for v, weight in neighbors.items():
                graph.add_vertex(v)
                graph.add_edge(u, v, weight)
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
            if len(tokens) >= 3:
                try:
                    f_val = float(tokens[2])
                    weight = int(f_val) if f_val.is_integer() else f_val
                except ValueError:
                    weight = 1
            else:
                weight = 1
            graph.add_edge(u, v, weight)
        return graph

    def twinport_description(self):
        """
        Build a TWINPORT-encoded graph description (Vanilla Twin-Port).

        This version of TWINPORT keeps the Twin-Port Edge Tickets and Local Incidence 
        Ledgers, but drops the ECC-Anchor Node Encoding in favor of using the original 
        node names. This allows deterministic extraction and evaluation of the adjacency 
        matrix by the LLM.

        1. Twin-Port Edge Tickets: every edge is expressed as two endpoint-local port
           addresses (half-edges). This turns each edge into an explicitly addressable
           incidence object, enabling pointer-like multi-hop traversal.

        2. Local Incidence Ledger: every node gets a per-port listing that shows 
           which edge ticket arrives or departs at each port, creating a redundant 
           but internally checkable graph representation.
        """
        vertices = sorted(self.vertices, key=Graph._natural_sort_key)
        graph_type = "DIRECTED" if self.directed else "UNDIRECTED"
        num_edges = self.get_number_of_edges()

        # ── Step 1: assign random topology-independent port numbers ───────────────
        port_counter = {v: 1 for v in vertices}
        edge_tickets = []
        ledger = {v: [] for v in vertices}
        seen_undirected = set()

        edge_num = 1
        for u in vertices:
            neighbors = sorted(self.adjacency.get(u, {}).items(),
                               key=lambda kv: Graph._natural_sort_key(kv[0]))
            for v, weight in neighbors:
                if not self.directed:
                    pair = tuple(sorted((u, v), key=Graph._natural_sort_key))
                    if pair in seen_undirected:
                        continue
                    seen_undirected.add(pair)

                # Assign local port numbers
                port_u = port_counter[u]
                port_counter[u] += 1
                port_v = port_counter[v]
                port_counter[v] += 1

                edge_id = f"E{edge_num:02d}"
                edge_num += 1

                w_str = f"+{weight:.3f}" if weight >= 0 else f"{weight:.3f}"

                if self.directed:
                    ticket = (
                        f"EDGE: {edge_id}  "
                        f"TAIL_NODE: {u}  TAIL_PORT: p{port_u:02d}  "
                        f"HEAD_NODE: {v}  HEAD_PORT: p{port_v:02d}  "
                        f"WEIGHT: {w_str}"
                    )
                    ledger[u].append(f"  OUT p{port_u:02d} -> {v}  [{edge_id}]")
                    ledger[v].append(f"  IN  p{port_v:02d} <- {u}  [{edge_id}]")
                else:
                    ticket = (
                        f"EDGE: {edge_id}  "
                        f"NODE1: {u}  PORT1: p{port_u:02d}  "
                        f"NODE2: {v}  PORT2: p{port_v:02d}  "
                        f"WEIGHT: {w_str}"
                    )
                    ledger[u].append(f"  UNDIR p{port_u:02d} <-> {v}  [{edge_id}]")
                    ledger[v].append(f"  UNDIR p{port_v:02d} <-> {u}  [{edge_id}]")

                edge_tickets.append(ticket)

        # ── Step 2: assemble the output string ────────────────────────────────────
        lines = []
        lines.append(f"GRAPH TYPE: {graph_type}")
        lines.append("")
        lines.append("INCIDENCE LEDGER:")
        for v in vertices:
            lines.append(f"\n  NODE {v}:")
            if ledger[v]:
                for entry in ledger[v]:
                    lines.append(entry)
            else:
                lines.append("    (isolated)")
        lines.append("")
        lines.append("EDGE TICKETS:")
        for ticket in edge_tickets:
            lines.append(f"  {ticket}")
        lines.append("")

        return "\n".join(lines)

    def incident_description(self):
        """Build the natural-language graph description directly from a Graph object using incident encoding method."""
        vertices = sorted(self.vertices, key=Graph._natural_sort_key)
        graph_type = "DIRECTED" if self.directed else "UNDIRECTED"
        vertex_word = "vertex" if len(vertices) == 1 else "vertices"

        description = (
            f"Graph: a {graph_type.lower()} weighted graph with {len(vertices)} {vertex_word}. "
            f"The vertices are {', '.join(vertices)}.\n"
        )
        num_edges = self.get_number_of_edges()
        description += f"The graph contains {num_edges} {'edge' if num_edges == 1 else 'edges'}.\n"
        description += "Edge information:\n"

        if self.directed:
            adjacency = self.adjacency
            for vertex in vertices:
                outgoing = sorted(adjacency.get(vertex, {}).items(), key=lambda item: Graph._natural_sort_key(item[0]))
                if not outgoing:
                    description += f"- Vertex {vertex} has no outgoing edges.\n"
                    continue

                edge_descriptions = [f"{neighbor} with weight {weight}" for neighbor, weight in outgoing]
                if len(edge_descriptions) == 1:
                    description += f"- Vertex {vertex} has an outgoing edge to {edge_descriptions[0]}.\n"
                else:
                    edge_text = ", ".join(edge_descriptions[:-1]) + f", and {edge_descriptions[-1]}"
                    description += f"- Vertex {vertex} has outgoing edges to {edge_text}.\n"
        else:
            seen = set()
            neighbors = {vertex: [] for vertex in vertices}
            for u, v, weight in self.get_edge_list():
                pair = tuple(sorted((u, v)))
                if pair in seen:
                    continue
                seen.add(pair)
                neighbors[u].append((v, weight))
                neighbors[v].append((u, weight))

            for vertex in vertices:
                incident = sorted(neighbors.get(vertex, []), key=lambda item: Graph._natural_sort_key(item[0]))
                if not incident:
                    description += f"- Vertex {vertex} is not connected to any other vertex.\n"
                    continue

                edge_descriptions = [f"{neighbor} with weight {weight}" for neighbor, weight in incident]
                if len(edge_descriptions) == 1:
                    description += f"- Vertex {vertex} is connected to {edge_descriptions[0]}.\n"
                else:
                    edge_text = ", ".join(edge_descriptions[:-1]) + f", and {edge_descriptions[-1]}"
                    description += f"- Vertex {vertex} is connected to {edge_text}.\n"

        return description