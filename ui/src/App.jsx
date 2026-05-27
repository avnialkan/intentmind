import { useState, useEffect, useRef, useCallback } from "react";

// ─── STYLES & THEMING ─────────────────────────────────────────────────────────
const COLORS = {
  bg1: "#212121",
  bg2: "#171717",
  bg3: "#2f2f2f",
  panelBg: "#2f2f2f",
  border: "#424242",
  accent: "#10a37f",
  accentGlow: "rgba(16, 163, 127, 0.2)",
  accent2: "#8e8ea0",
  text: "#ececec",
  muted: "#9b9b9b",
  green: "#10a37f",
  orange: "#f59e0b",
  red: "#ef4444",
};

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function ProPanel({ children, style }) {
  return (
    <div style={{
      background: COLORS.bg1,
      border: `1px solid ${COLORS.border}`,
      borderRadius: "12px",
      ...style
    }}>
      {children}
    </div>
  );
}

function EnergyBar({ energy }) {
  const pct = Math.round(energy * 100);
  const color = energy > 0.6 ? COLORS.green : energy > 0.35 ? COLORS.accent : COLORS.muted;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
      <div style={{
        flex: 1, height: 4, background: "rgba(255,255,255,0.05)", 
        borderRadius: 2, overflow: "hidden"
      }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: color,
          boxShadow: `0 0 8px ${color}`,
          transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1), background 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: 10, color: COLORS.text, fontFamily: "monospace", width: 24, textAlign: "right" }}>
        {energy.toFixed(2)}
      </span>
    </div>
  );
}

function StatCard({ title, value, unit, icon, color }) {
  return (
    <div style={{
      background: COLORS.panelBg,
      border: `1px solid ${COLORS.border}`,
      borderRadius: "8px",
      padding: "16px",
      display: "flex",
      flexDirection: "column",
      gap: 8,
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, color: COLORS.muted, fontWeight: 600, letterSpacing: 0.5 }}>{title}</span>
        <span style={{ fontSize: 14 }}>{icon}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ fontSize: 24, fontWeight: 600, color: COLORS.text, fontFamily: "system-ui, sans-serif" }}>{value}</span>
        {unit && <span style={{ fontSize: 12, color: COLORS.muted }}>{unit}</span>}
      </div>
    </div>
  );
}

function CognitiveGraph({ nodes, activeIds, onNodeSelect }) {
  const canvasRef = useRef(null);
  const simRef = useRef({ nodes: [], edges: [], hovered: null, dragging: null });
  const animRef = useRef(null);

  // Build simulation data whenever nodes change
  useEffect(() => {
    const sim = simRef.current;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width = canvas.offsetWidth * 2;
    const H = canvas.height = canvas.offsetHeight * 2;
    const cx = W / 2, cy = H / 2;

    // Preserve positions for existing nodes
    const oldPositions = {};
    sim.nodes.forEach(n => { oldPositions[n.id] = { x: n.x, y: n.y, vx: n.vx, vy: n.vy }; });

    // Build node list
    const simNodes = nodes.map((n, i) => {
      const old = oldPositions[n.id];
      const angle = (i / Math.max(nodes.length, 1)) * Math.PI * 2;
      return {
        ...n,
        x: old ? old.x : cx + Math.cos(angle) * (H * 0.25) + (Math.random() - 0.5) * 40,
        y: old ? old.y : cy + Math.sin(angle) * (H * 0.25) + (Math.random() - 0.5) * 40,
        vx: old ? old.vx * 0.5 : 0,
        vy: old ? old.vy * 0.5 : 0,
        radius: Math.max(12, Math.min(36, n.energy * 40)) * 2,
        isActive: activeIds.includes(n.id),
      };
    });

    // Build edge list from node edge data
    const simEdges = [];
    const nodeMap = {};
    simNodes.forEach(n => { nodeMap[n.id] = n; });
    nodes.forEach(n => {
      (n.edges || []).forEach(e => {
        if (nodeMap[n.id] && nodeMap[e.target]) {
          simEdges.push({
            source: nodeMap[n.id],
            target: nodeMap[e.target],
            weight: e.weight || 0.5,
            confidence: e.confidence || 0.5,
          });
        }
      });
    });

    sim.nodes = simNodes;
    sim.edges = simEdges;
    sim.W = W;
    sim.H = H;
  }, [nodes, activeIds]);

  // Force simulation + render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const sim = simRef.current;

    function tick() {
      const { nodes: sn, edges: se, W, H } = sim;
      if (!sn.length) {
        // Draw empty state
        ctx.clearRect(0, 0, W || 1, H || 1);
        ctx.fillStyle = "rgba(148,163,184,0.3)";
        ctx.font = "24px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Send a message to build the graph", (W || 600) / 2, (H || 600) / 2);
        animRef.current = requestAnimationFrame(tick);
        return;
      }
      const cx = W / 2, cy = H / 2;

      // --- Forces ---
      // Center gravity
      sn.forEach(n => {
        n.vx += (cx - n.x) * 0.0008;
        n.vy += (cy - n.y) * 0.0008;
      });

      // Repulsion
      for (let i = 0; i < sn.length; i++) {
        for (let j = i + 1; j < sn.length; j++) {
          let dx = sn[j].x - sn[i].x;
          let dy = sn[j].y - sn[i].y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          let force = 18000 / (dist * dist);
          let fx = (dx / dist) * force;
          let fy = (dy / dist) * force;
          sn[i].vx -= fx; sn[i].vy -= fy;
          sn[j].vx += fx; sn[j].vy += fy;
        }
      }

      // Spring (edges)
      se.forEach(e => {
        let dx = e.target.x - e.source.x;
        let dy = e.target.y - e.source.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let ideal = 120 + (1 - e.weight) * 80;
        let force = (dist - ideal) * 0.004 * e.weight;
        let fx = (dx / dist) * force;
        let fy = (dy / dist) * force;
        e.source.vx += fx; e.source.vy += fy;
        e.target.vx -= fx; e.target.vy -= fy;
      });

      // Velocity & position update
      sn.forEach(n => {
        if (sim.dragging === n.id) return;
        n.vx *= 0.88;
        n.vy *= 0.88;
        n.x += n.vx;
        n.y += n.vy;
        // Boundary
        n.x = Math.max(n.radius, Math.min(W - n.radius, n.x));
        n.y = Math.max(n.radius, Math.min(H - n.radius, n.y));
      });

      // --- Render ---
      ctx.clearRect(0, 0, W, H);

      // Edges
      se.forEach(e => {
        const srcActive = e.source.isActive;
        const tgtActive = e.target.isActive;
        const highlight = srcActive && tgtActive;
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.strokeStyle = highlight
          ? `rgba(99, 102, 241, ${0.6 + e.confidence * 0.4})`
          : `rgba(148, 163, 184, ${0.35 + e.confidence * 0.35})`;
        ctx.lineWidth = highlight ? (4 + e.weight * 6) : (1.5 + e.weight * 4);
        ctx.stroke();

        // Weight label on highlighted edges
        if (highlight) {
          const mx = (e.source.x + e.target.x) / 2;
          const my = (e.source.y + e.target.y) / 2;
          ctx.fillStyle = "rgba(99, 102, 241, 0.7)";
          ctx.font = "18px monospace";
          ctx.textAlign = "center";
          ctx.fillText(e.weight.toFixed(2), mx, my - 6);
        }
      });

      // Nodes
      sn.forEach(n => {
        const isHovered = sim.hovered === n.id;

        // Glow
        if (n.isActive || isHovered) {
          const grd = ctx.createRadialGradient(n.x, n.y, n.radius * 0.5, n.x, n.y, n.radius * 2.5);
          const glowColor = n.isActive ? "16, 163, 127" : "142, 142, 160";
          grd.addColorStop(0, `rgba(${glowColor}, 0.3)`);
          grd.addColorStop(1, `rgba(${glowColor}, 0)`);
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = grd;
          ctx.fill();
        }

        // Circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        if (n.isActive) {
          const grd = ctx.createRadialGradient(n.x - n.radius * 0.3, n.y - n.radius * 0.3, 0, n.x, n.y, n.radius);
          grd.addColorStop(0, "#10b981");
          grd.addColorStop(1, "#059669");
          ctx.fillStyle = grd;
        } else {
          // Heatmap coloring based on energy (0.0 to 1.0+)
          const heat = Math.min(1, Math.max(0, n.energy));
          // Cold: #1e3a8a (blue), Hot: #f59e0b (orange/yellow)
          const r = Math.round(30 + heat * 215);
          const g = Math.round(58 + heat * 100);
          const b = Math.round(138 - heat * 127);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + heat * 0.5})`;
        }
        ctx.fill();
        ctx.strokeStyle = n.isActive ? "rgba(129, 140, 248, 0.6)" : "rgba(255,255,255,0.1)";
        ctx.lineWidth = isHovered ? 4 : 2;
        ctx.stroke();

        // Label
        const label = n.id.length > 18 ? n.id.slice(0, 16) + "…" : n.id;
        ctx.font = `${n.isActive ? "bold " : ""}${n.isActive ? 22 : 18}px Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillStyle = n.isActive ? "#f8fafc" : `rgba(248, 250, 252, ${0.5 + n.energy * 0.4})`;
        ctx.fillText(label, n.x, n.y + n.radius + 22);

        // Energy value
        ctx.font = "16px monospace";
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.fillText(n.energy.toFixed(2), n.x, n.y + 6);
      });

      // Tooltip for hovered node
      if (sim.hovered) {
        const hn = sn.find(n => n.id === sim.hovered);
        if (hn) {
          const lines = [
            hn.id,
            `Energy: ${hn.energy.toFixed(3)}`,
            `Hits: ${hn.hitCount || 0}`,
            `Edges: ${(hn.edges || []).length}`,
          ];
          const tw = 280, th = lines.length * 30 + 16;
          let tx = hn.x + hn.radius + 20, ty = hn.y - th / 2;
          if (tx + tw > W) tx = hn.x - hn.radius - tw - 20;
          if (ty < 10) ty = 10;
          ctx.fillStyle = "rgba(10, 10, 30, 0.92)";
          ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.roundRect(tx, ty, tw, th, 12);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#f8fafc";
          ctx.font = "bold 20px Inter, sans-serif";
          ctx.textAlign = "left";
          ctx.fillText(lines[0], tx + 16, ty + 28);
          ctx.font = "18px monospace";
          ctx.fillStyle = "#94a3b8";
          lines.slice(1).forEach((l, i) => {
            ctx.fillText(l, tx + 16, ty + 28 + (i + 1) * 26);
          });
        }
      }

      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // Mouse interaction
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const sim = simRef.current;
    const scale = 2; // canvas pixel ratio

    function getNode(e) {
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * scale;
      const my = (e.clientY - rect.top) * scale;
      for (const n of sim.nodes) {
        const dx = mx - n.x, dy = my - n.y;
        if (dx * dx + dy * dy < (n.radius + 10) * (n.radius + 10)) return n;
      }
      return null;
    }

    function onMove(e) {
      const n = getNode(e);
      sim.hovered = n ? n.id : null;
      canvas.style.cursor = n ? "grab" : "default";
      if (sim.dragging) {
        const rect = canvas.getBoundingClientRect();
        const dn = sim.nodes.find(nd => nd.id === sim.dragging);
        if (dn) {
          dn.x = (e.clientX - rect.left) * scale;
          dn.y = (e.clientY - rect.top) * scale;
          dn.vx = 0; dn.vy = 0;
        }
      }
    }
    function onDown(e) {
      const n = getNode(e);
      if (n) { sim.dragging = n.id; canvas.style.cursor = "grabbing"; }
    }
    let isDragging = false;
    function onMove(e) {
      if (sim.dragging) isDragging = true;
      const n = getNode(e);
      sim.hovered = n ? n.id : null;
      canvas.style.cursor = n ? (sim.dragging ? "grabbing" : "pointer") : "default";
      if (sim.dragging) {
        const rect = canvas.getBoundingClientRect();
        const dn = sim.nodes.find(nd => nd.id === sim.dragging);
        if (dn) {
          dn.x = (e.clientX - rect.left) * scale;
          dn.y = (e.clientY - rect.top) * scale;
          dn.vx = 0; dn.vy = 0;
        }
      }
    }
    function onDown(e) {
      isDragging = false;
      const n = getNode(e);
      if (n) { sim.dragging = n.id; canvas.style.cursor = "grabbing"; }
    }
    function onUp(e) { 
      if (!isDragging && sim.dragging === null) {
        // This was a click, not a drag release.
        const n = getNode(e);
        if (n && onNodeSelect) {
          onNodeSelect(n);
        } else if (!n && onNodeSelect) {
          onNodeSelect(null);
        }
      }
      sim.dragging = null; 
      isDragging = false;
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", onUp);
    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("mouseleave", onUp);
    };
  }, []);

  return (
    <ProPanel style={{ display: "flex", flexDirection: "column", height: "100%", padding: 0, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "16px 20px", borderBottom: `1px solid ${COLORS.border}` }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS.accent, boxShadow: `0 0 10px ${COLORS.accent}` }} />
        <span style={{ fontSize: 11, color: COLORS.text, letterSpacing: 2, fontWeight: 600 }}>COGNITIVE GRAPH</span>
        <span style={{ fontSize: 10, color: COLORS.muted, marginLeft: "auto" }}>{nodes.length} nodes</span>
      </div>
      <canvas ref={canvasRef} style={{ flex: 1, width: "100%", display: "block" }} />
    </ProPanel>
  );
}

function CognitivePath({ path }) {
  if (!path || (!path.seeds?.length && !path.items?.length)) return null;

  const [expanded, setExpanded] = useState(false);

  const layerLabel = (layer) => {
    if (layer === 0 || layer === "direct") return "DIRECT";
    if (layer === 1 || layer === "associated") return "ASSOCIATED";
    if (layer === 2 || layer === "weak_echo") return "WEAK ECHO";
    return String(layer).toUpperCase();
  };

  const layerColor = (layer) => {
    if (layer === 0 || layer === "direct") return COLORS.green;
    if (layer === 1 || layer === "associated") return COLORS.accent;
    if (layer === 2 || layer === "weak_echo") return COLORS.orange;
    return COLORS.muted;
  };

  return (
    <div style={{
      marginTop: 8, padding: "12px 16px",
      background: "rgba(255,255,255,0.02)",
      border: `1px solid rgba(255,255,255,0.06)`,
      borderRadius: "12px",
      fontSize: 11, color: COLORS.muted,
      maxWidth: "85%",
    }}>
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          cursor: "pointer", userSelect: "none",
        }}
      >
        <span style={{ color: COLORS.accent, fontSize: 13 }}>⟁</span>
        <span style={{ fontWeight: 600, letterSpacing: 1, color: COLORS.accent, fontSize: 10 }}>
          COGNITIVE PATH
        </span>
        <span style={{ fontSize: 10, color: COLORS.muted, marginLeft: 4 }}>
          {path.memories_used || 0} memories · {path.seeds?.length || 0} seeds
        </span>
        <span style={{ marginLeft: "auto", fontSize: 10, transition: "transform 0.2s", transform: expanded ? "rotate(90deg)" : "rotate(0)" }}>▸</span>
      </div>

      {/* Seeds line */}
      {path.seeds?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 8 }}>
          {path.seeds.map((s, i) => (
            <span key={i} style={{
              padding: "2px 8px", borderRadius: 8,
              background: "rgba(99, 102, 241, 0.15)",
              border: "1px solid rgba(99, 102, 241, 0.25)",
              color: COLORS.accent, fontSize: 10, fontWeight: 500,
            }}>
              🌱 {typeof s === "string" ? s : s.label || s.intent || JSON.stringify(s)}
            </span>
          ))}
          {path.resonant?.length > 0 && path.resonant.slice(0, 5).map((r, i) => (
            <span key={`r-${i}`} style={{
              padding: "2px 8px", borderRadius: 8,
              background: "rgba(168, 85, 247, 0.1)",
              border: "1px solid rgba(168, 85, 247, 0.2)",
              color: COLORS.accent2, fontSize: 10, fontWeight: 500,
            }}>
              ◎ {typeof r === "string" ? r : r.label || r.intent || JSON.stringify(r)}
            </span>
          ))}
        </div>
      )}

      {/* Expanded: memory path details */}
      {expanded && path.items?.length > 0 && (
        <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
          {path.items.map((item, i) => (
            <div key={i} style={{
              display: "flex", flexDirection: "column", gap: 2,
              padding: "8px 10px",
              background: "rgba(0,0,0,0.2)",
              borderLeft: `3px solid ${layerColor(item.layer)}`,
              borderRadius: "0 8px 8px 0",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                <span style={{
                  fontSize: 9, padding: "1px 6px", borderRadius: 6,
                  background: layerColor(item.layer), color: "#000", fontWeight: 700,
                }}>
                  {layerLabel(item.layer)}
                </span>
                <span style={{ fontWeight: 600, color: COLORS.text, fontSize: 11 }}>{item.intent}</span>
                <span style={{ fontSize: 10, color: COLORS.muted }}>score: {item.score}</span>
                {item.called_by && (
                  <span style={{ fontSize: 10, color: COLORS.accent2 }}>← {item.called_by}</span>
                )}
              </div>
              {item.path?.length > 0 && (
                <div style={{ fontSize: 10, color: COLORS.accent, marginTop: 2 }}>
                  {item.path.join(" → ")}
                </div>
              )}
              <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", marginTop: 2, fontStyle: "italic" }}>
                "{item.text}"
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ role, content, cognitivePath }) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      marginBottom: 24,
      animation: "fadeSlideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards",
    }}>
      <div style={{
        maxWidth: "85%",
        background: isUser ? COLORS.bg3 : "transparent",
        border: "none",
        backdropFilter: "none",
        borderRadius: "12px",
        padding: "16px 20px",
        fontSize: 14,
        color: COLORS.text,
        lineHeight: 1.6,
        whiteSpace: "pre-wrap",
        boxShadow: "none",
      }}>
        {content}
      </div>
      <div style={{ 
        fontSize: 10, color: COLORS.muted, marginTop: 8, 
        padding: "0 8px", display: "flex", gap: 6, alignItems: "center" 
      }}>
        <div style={{ width: 4, height: 4, borderRadius: "50%", background: isUser ? COLORS.accent : COLORS.accent2 }} />
        {isUser ? "User" : "Intentmind"}
      </div>
      {!isUser && cognitivePath && <CognitivePath path={cognitivePath} />}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [field, setField] = useState({ 
    nodes: [], 
    active_ids: [], 
    stats: { total_nodes: 0, total_edges: 0, avg_energy: 0, total_chunks: 0 } 
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setLoading(true);

    const userMsg = { role: "user", content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages }),
      });
      if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
      
      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.response, cognitivePath: data.cognitive_path }]);
      setField(data.field);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "Connection Error: " + e.message }]);
    }

    setLoading(false);
  }, [input, loading, messages]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  // Safe fallback for stats
  const stats = field.stats || { total_nodes: 0, total_edges: 0, avg_energy: 0, total_chunks: 0 };
  const graphDensity = stats.total_nodes > 1 ? ((stats.total_edges) / (stats.total_nodes * (stats.total_nodes - 1))) * 100 : 0;

  return (
    <div className="app-container">
      <div style={{
        display: "flex", height: "100vh", width: "100vw",
        boxSizing: "border-box", margin: 0, padding: 0
      }}>
        
        {/* MAIN COLUMN: Chat Interface */}
        <div style={{ 
          flex: 1, display: "flex", flexDirection: "column", 
          overflow: "hidden", position: "relative", alignItems: "center"
        }}>
          <div style={{ width: "100%", maxWidth: 800, flex: 1, display: "flex", flexDirection: "column" }}>
          {/* Header */}
          <div style={{
            padding: "20px 24px", borderBottom: `1px solid ${COLORS.border}`,
            display: "flex", alignItems: "center", justifyContent: "space-between",
            background: "rgba(0,0,0,0.2)"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 10, height: 10, borderRadius: "50%",
                background: COLORS.green, boxShadow: `0 0 12px ${COLORS.green}`,
              }} />
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: 0.5, color: COLORS.text }}>Intentmind Core</div>
                <div style={{ fontSize: 11, color: COLORS.muted }}>v0.3.0 (Cognitive Layer)</div>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <div style={{ flex: 1, overflowY: "auto", padding: "32px 32px" }} className="hide-scrollbar">
            {messages.length === 0 && (
              <div style={{
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                height: "100%", color: COLORS.muted, opacity: 0.6
              }}>
                <div style={{ fontSize: 48, marginBottom: 16, filter: "drop-shadow(0 0 10px rgba(255,255,255,0.2))" }}>✺</div>
                <div style={{ fontSize: 16, fontWeight: 500, letterSpacing: 1 }}>SYSTEM READY</div>
                <div style={{ fontSize: 12, marginTop: 8, maxWidth: 300, textAlign: "center", lineHeight: 1.6 }}>
                  Query the knowledge graph. Real-time cognitive tracking enabled.
                </div>
              </div>
            )}
            
            {messages.map((m, i) => <Message key={i} {...m} />)}
            
            {loading && (
              <div style={{ 
                display: "flex", alignItems: "center", gap: 12, 
                color: COLORS.muted, fontSize: 13, padding: "16px 20px" 
              }}>
                <div className="typing-indicator"><span></span><span></span><span></span></div>
                processing intents...
              </div>
            )}
            <div ref={bottomRef} style={{ height: 1 }} />
          </div>

          {/* Input Area */}
          <div style={{
            padding: "24px 32px", borderTop: `1px solid ${COLORS.border}`,
            background: "rgba(0,0,0,0.2)"
          }}>
            <div style={{
              display: "flex", gap: 16, background: "rgba(0,0,0,0.3)",
              border: `1px solid rgba(255,255,255,0.1)`, borderRadius: "24px",
              padding: "8px 12px", alignItems: "flex-end",
              transition: "border 0.3s",
            }} className="input-wrapper">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask something to explore the graph..."
                rows={1}
                style={{
                  flex: 1, background: "transparent", border: "none",
                  padding: "12px 16px", color: COLORS.text,
                  fontSize: 15, resize: "none", outline: "none",
                  fontFamily: "inherit", maxHeight: 120, lineHeight: 1.5
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = (e.target.scrollHeight) + "px";
                }}
              />
              <button
                onClick={send}
                disabled={loading || !input.trim()}
                style={{
                  background: loading || !input.trim() ? "rgba(255,255,255,0.05)" : COLORS.accent,
                  color: loading || !input.trim() ? COLORS.muted : "#fff",
                  border: "none", borderRadius: "18px", width: 44, height: 44,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: loading ? "wait" : "pointer", transition: "all 0.3s",
                  marginBottom: 2
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
          </div>
          </div>
        </div>

        {/* RIGHT SIDEBAR: Telemetry & Graph */}
        <div style={{ flex: "0 0 380px", display: "flex", flexDirection: "column", gap: 16, padding: 20, background: COLORS.bg2, borderLeft: `1px solid ${COLORS.border}`, overflowY: "auto" }}>
          
          <div style={{ display: "flex", alignItems: "center", gap: 8, paddingBottom: 8, borderBottom: `1px solid ${COLORS.border}` }}>
            <span style={{ fontSize: 12, color: COLORS.muted, letterSpacing: 1, fontWeight: 600 }}>COGNITIVE TELEMETRY</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <StatCard title="MEMORY NODES" value={stats.total_nodes} icon="⎈" />
            <StatCard title="ACTIVE EDGES" value={stats.total_edges} icon="⚄" />
            <StatCard title="SEMANTIC LEAPS" value={Math.floor(stats.total_edges * 0.3)} icon="⚡" />
            <StatCard title="CONSOLIDATION" value={((stats.total_nodes > 0 ? stats.total_chunks / stats.total_nodes : 0) * 100).toFixed(1)} unit="%" icon="🧬" />
          </div>

          <div style={{ flex: "0 0 300px", marginTop: 8 }}>
            <CognitiveGraph nodes={field.nodes} activeIds={field.active_ids} onNodeSelect={setSelectedNode} />
          </div>

        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        body { margin: 0; background: ${COLORS.bg1}; color: ${COLORS.text}; font-family: 'Inter', sans-serif; overflow: hidden; }
        
        .app-container { position: relative; height: 100vh; width: 100vw; overflow: hidden; }
        
        
          to { transform: scaleY(1.2); }
        }

        /* Custom Scrollbar */
        .hide-scrollbar::-webkit-scrollbar { width: 6px; }
        .hide-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .hide-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        .hide-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

        .input-wrapper:focus-within { border-color: ${COLORS.accent} !important; box-shadow: 0 0 0 1px ${COLORS.accent}; }

        /* Typing Indicator */
        .typing-indicator { display: flex; gap: 4px; }
        .typing-indicator span {
          width: 6px; height: 6px; background: ${COLORS.muted}; border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); background: ${COLORS.accent}; } }
      `}</style>
    </div>
  );
}
