from __future__ import annotations
import os

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError:
    nx = None
    Network = None

class GraphVisualizer:
    def __init__(self, store):
        self.store = store

    def visualize(self, output_html: str = "memory_map.html") -> str:
        if nx is None or Network is None:
            raise ImportError("Görselleştirme için 'networkx' ve 'pyvis' kütüphaneleri gereklidir. (pip install networkx pyvis)")

        G = nx.Graph()

        # Add intents as nodes
        for intent in self.store.intents.values():
            if intent.state == "archived":
                continue
                
            color = "#4CAF50" # Default Active
            if intent.state == "weak":
                color = "#FFC107"
            
            size = max(10, int(intent.energy * 30))
            
            G.add_node(
                intent.intent_id,
                label=intent.label,
                title=f"Type: {intent.type}\nEnergy: {intent.energy:.2f}\nState: {intent.state}",
                color=color,
                size=size,
                font={"color": "white" if color == "#4CAF50" else "black"}
            )

        # Add edges
        for edge in self.store.edges.values():
            if edge.state == "archived":
                continue
            if edge.source_id not in self.store.intents or edge.target_id not in self.store.intents:
                continue
                
            color = "#9E9E9E" # Default Active
            if edge.state == "candidate":
                color = "#2196F3"
            elif edge.state == "weak":
                color = "#FF9800"
                
            width = max(1, int(edge.weight * 5))
            
            G.add_edge(
                edge.source_id,
                edge.target_id,
                title=f"Type: {edge.edge_type}\nWeight: {edge.weight:.2f}\nEnergy: {edge.energy:.2f}",
                color=color,
                width=width
            )

        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)
        net.from_nx(G)
        
        # Disable physics for large graphs to prevent infinite jiggling, or use Barnes Hut
        net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

        # Save HTML
        net.save_graph(output_html)
        return os.path.abspath(output_html)
