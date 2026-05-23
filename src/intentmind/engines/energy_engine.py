import time
import math
from typing import List, Dict
from ..store import MemoryStore
from ..models import IntentNode

LAMBDA = 0.006
THRESHOLD = 0.18
BOOST = 0.45

class EnergyEngine:
    def __init__(self, store: MemoryStore):
        self.store = store

    def activate_field(self, seed_intent_ids: List[str], current_time: float | None = None) -> dict:
        """
        Run one query-time cognitive field step.

        Seed intents are the direct concepts activated by the user query. They
        are reinforced first, then energy is propagated to graph neighbors.
        The returned summary is deliberately serializable so it can be exposed
        in API responses, traces, and prompt context.
        """
        now = current_time or time.time()
        seed_ids = []
        seen = set()
        for intent_id in seed_intent_ids:
            if intent_id in self.store.intents and intent_id not in seen:
                seed_ids.append(intent_id)
                seen.add(intent_id)

        if not seed_ids:
            return {
                "seed_intents": [],
                "activated_intents": [],
                "gates": {},
            }
            
        # 1. Topic Boundary Detection & Suppression Engine
        snap = self.snapshot_all(now)
        highly_active_ids = {nid for nid, e in snap.items() if e > 0.4 and nid not in seed_ids}
        
        if highly_active_ids and seed_ids:
            # Check if new seeds are connected to the current active field
            is_connected = False
            for sid in seed_ids:
                if snap.get(sid, 0.0) > 0.3:
                    is_connected = True
                    break
                # Check 1-hop neighbors
                neighbors = [n_id for n_id, _ in self.store.get_neighbors(sid)]
                if highly_active_ids.intersection(neighbors):
                    is_connected = True
                    break
                    
            if not is_connected:
                # Topic Shift Detected! Apply Cognitive Suppression (Inhibition)
                for nid in highly_active_ids:
                    node = self.store.intents[nid]
                    # Penalty: suppress unrelated active field
                    self.write_energy(node, node.energy * 0.35, now)

        self.reinforce_nodes(seed_ids, seed_ids, now)
        activated_nodes = self.field_step(seed_ids, now)
        gates = self.compute_gates(seed_ids, now)

        seed_set = set(seed_ids)
        
        # 2. Active Intent Arena Limits (MAX_PRIMARY)
        # Sort nodes by energy
        sorted_nodes = sorted(activated_nodes, key=lambda n: n.energy, reverse=True)
        max_primary = 5
        primary_count = 0
        
        for node in sorted_nodes:
            if node.energy > 0.70:
                primary_count += 1
                if primary_count > max_primary and node.intent_id not in seed_set:
                    # Suppress overflow nodes
                    self.write_energy(node, node.energy * 0.65, now)

        activated = []
        for node in sorted_nodes:
            if node.intent_id not in seed_set:
                node.activation_count += 1
            activated.append({
                "intent_id": node.intent_id,
                "label": node.label,
                "energy": round(node.energy, 4),
                "state": node.state,
                "role": "seed" if node.intent_id in seed_set else "resonant",
                "gate": round(gates.get(node.intent_id, 0.0), 4),
            })

        return {
            "seed_intents": [
                {
                    "intent_id": intent_id,
                    "label": self.store.intents[intent_id].label,
                    "energy": round(self.store.intents[intent_id].energy, 4),
                }
                for intent_id in seed_ids
            ],
            "activated_intents": activated,
            "gates": {intent_id: round(value, 4) for intent_id, value in gates.items()},
        }

    def get_energy(self, node: IntentNode, current_time: float) -> float:
        """Calculate the continuous exponentially decayed energy of a node."""
        # Fix dt in hours for the decay calculation to match the UI behavior over real time,
        # or use seconds if simulation. We'll use hours to prevent instant loss.
        # But for the UI demo, they used seconds. Let's use minutes as a good middle ground for Python backend.
        # Let's just use raw seconds since last_active for the demo effect!
        dt = current_time - node.last_active
        if dt < 0:
            dt = 0
        # If we use raw seconds, LAMBDA=0.08 means it decays by half in ~8 seconds.
        # This is great for a realtime Cognitive Field demo!
        return node.energy * math.exp(-LAMBDA * dt)

    def write_energy(self, node: IntentNode, e: float, current_time: float):
        node.energy = e
        node.last_active = current_time

    def snapshot_all(self, current_time: float) -> Dict[str, float]:
        snap = {}
        for nid, node in self.store.intents.items():
            snap[nid] = self.get_energy(node, current_time)
        return snap

    def field_step(self, active_ids: List[str], current_time: float) -> List[IntentNode]:
        """
        Propagate energy through the cognitive resonance field.
        active_ids: List of newly activated or currently active intent IDs.
        """
        snap = self.snapshot_all(current_time)
        updates = {}
        
        # 1. Dynamic Supernode Scoring
        degrees = {nid: len(self.store.get_neighbors(nid)) for nid in self.store.intents}
        supernode_penalties = {}
        for nid, deg in degrees.items():
            # Penalty represents how much transmission is reduced. 
            # Transmission factor = 1.0 / math.log(2 + deg)
            # Penalty = 1.0 - transmission_factor
            transmission_factor = 1.0 / math.log(2 + deg)
            supernode_penalties[nid] = 1.0 - transmission_factor
        self.last_supernode_penalties = supernode_penalties
        
        # 2. Propagate energy from active nodes to neighbors
        for nid in active_ids:
            if nid not in self.store.intents:
                continue
            pe = snap.get(nid, 0.0)
            
            deg = degrees.get(nid, 0)
            transmission_factor = 1.0 / math.log(2 + deg)
            
            # Find neighbors via edges
            for edge in self.store.edges.values():
                if edge.source_id == nid:
                    target = edge.target_id
                    updates[target] = updates.get(target, 0.0) + (pe * edge.weight * BOOST * transmission_factor)
                elif edge.target_id == nid:
                    target = edge.source_id
                    updates[target] = updates.get(target, 0.0) + (pe * edge.weight * BOOST * transmission_factor)

        # Apply updates using the activation function. Query-time field
        # activation must not passively demote unrelated nodes; long embedder
        # calls would otherwise make later direct recalls disappear.
        new_active = []
        for nid, node in self.store.intents.items():
            base = snap.get(nid, 0.0)
            delta = updates.get(nid, 0.0)
            
            if delta > 0 or nid in active_ids:
                # Do not squash seed intents; they were just reinforced.
                if nid in active_ids:
                    ne = min(1.0, base + delta)
                else:
                    ne = (base + delta) / (1.0 + base + delta)
                self.write_energy(node, ne, current_time)
                
                if ne > THRESHOLD:
                    new_active.append(node)
                    node.state = "active"
                elif node.state != "active":
                    node.state = "weak"
                    
        return new_active

    def compute_gates(self, active_ids: List[str], current_time: float) -> Dict[str, float]:
        """Attention gating: nodes get more boost if their neighbors are also active."""
        snap = self.snapshot_all(current_time)
        vals = list(snap.values())
        global_mean = max(sum(vals) / len(vals) if vals else 0.0, 1e-6)
        
        active_set = set(active_ids)
        gates = {}
        
        for nid, node in self.store.intents.items():
            neighbor_energies = []
            for edge in self.store.edges.values():
                if edge.source_id == nid and edge.target_id in active_set:
                    neighbor_energies.append(snap.get(edge.target_id, 0.0))
                elif edge.target_id == nid and edge.source_id in active_set:
                    neighbor_energies.append(snap.get(edge.source_id, 0.0))
                    
            if neighbor_energies:
                fp = sum(neighbor_energies) / len(neighbor_energies)
            else:
                fp = 0.0
                
            gates[nid] = fp / global_mean
            
        return gates

    def reinforce_nodes(self, intent_ids: List[str], active_ids: List[str], current_time: float):
        """Boost nodes that were directly hit/matched. Direct hits get max energy (1.0)."""
        for nid in intent_ids:
            if nid in self.store.intents:
                node = self.store.intents[nid]
                # Seed intents are in active working memory, so they jump to max energy.
                self.write_energy(node, 1.0, current_time)
                node.activation_count += 1

    def tick(self, hours: float = 1.0) -> dict:
        """
        Backward compatibility for existing tick calls. 
        In this continuous model, ticking is passive, but we can do a cleanup.
        """
        now = time.time()
        # Ensure all nodes are updated with current decayed energy
        for node in self.store.intents.values():
            e = self.get_energy(node, now)
            self.write_energy(node, e, now)
            if e < 0.05:
                node.state = "archived"
        return {"timestamp": now}
