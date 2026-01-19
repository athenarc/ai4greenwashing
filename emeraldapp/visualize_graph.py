import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np
from pyvis.network import Network
np.random.seed(42)

def visualize_evidence_subgraph(
    evidence: list[dict[str, Any]], 
    company_name: str,
    height: str = "400px"
) -> str:
    """
    Generates an interactive PyVis HTML string for the evidence subgraph.
    Includes Custom Styled HTML Tooltips.
    """
    G = nx.DiGraph()
    
    color_map = {
        "Organization": "#95C469", "Person": "#96d82d",
        "Product": "#7a9a6e", "Material": "#8fbc8f", "Facility": "#9eb384",
        "KPIObservation": "#cedebd", "Emission": "#b5c9a8", "Waste": "#a3b996",
        "Goal": "#e8dcc5", "Plan": "#f0e6d2", "Initiative": "#faf1e4",
        "Standard": "#e6d7b9", "Regulation": "#dcc8a8", "Certification": "#d2b48c",
        "ThirdPartyVerification": "#d3d3d3", "Controversy": "#a9a9a9",
        "Penalty": "#BDBDBD", "MediaReport": "#c0c0c0",
        "Location": "#ffffff", "Country": "#f5f5f5", "Community": "#f9f9f9",
        "Authority": "#f0f0f0", "Investment": "#a0a0a0", "Project": "#b0b0b0",
        "CarbonOffsetProject": "#c0c0c0", "ScienceBasedTarget": "#d0d0d0",
        "Unknown": "#e0e0e0"
    }

    def create_tooltip(node_type, props):
        """
        Creates a simple multi-line plain text tooltip.
        PyVis will display this with line breaks preserved.
        """
        SYSTEM_KEYS = {
            'id', 'node_id', '_node_key', 'label', 'title', 'group', 'shape', 'color', 'font', 
            'size', 'x', 'y', 'physics', 'hidden', 'level', 'margin', 
            'embedding', 'valid_from', 'valid_to', 'is_current', 'type'
        }
        
        lines = [f"[{node_type}]", "─" * 20]
        
        for k, v in props.items():
            if k not in SYSTEM_KEYS and v is not None and str(v).strip() != "":
                val_str = str(v)
                if len(val_str) > 50:
                    val_str = val_str[:50] + "..."
                lines.append(f"{k}: {val_str}")
        
        if len(lines) == 2:  # Only header and separator
            lines.append("No additional properties")
        
        return "\n".join(lines)

    def get_label(node_type, props, fallback):
        priority = ["name", "title", "description", "category", "project_id"]
        for p in priority:
            if p in props and props[p]:
                return str(props[p])[:20] + ".." if len(str(props[p])) > 20 else str(props[p])
        return f"{node_type}"

    # Build Graph
    G.add_node(
        company_name, 
        label=company_name, 
        title=f"[Target Company]\n{'─' * 20}\n{company_name}", 
        color=color_map["Organization"], 
        shape="box"
    )
    added_nodes = {company_name}

    for item in evidence:
        node_id = item.get('node_id', str(np.random.randint(10000, 99999)))
        node_labels = item.get('labels', [])
        primary_label = node_labels[0] if node_labels else "Unknown"
        props = item.get('properties', {})
        rel_type = item.get('rel_type', 'RELATED')
        
        unique_id = f"{primary_label}_{node_id}"
        
        if unique_id not in added_nodes:
            col = color_map.get(primary_label, color_map["Unknown"])
            lbl = get_label(primary_label, props, node_id)
            
            # GENERATE PLAIN TEXT TOOLTIP
            tooltip = create_tooltip(primary_label, props)
            
            G.add_node(unique_id, label=lbl, title=tooltip, color=col, group=primary_label)
            added_nodes.add(unique_id)
        
        G.add_edge(company_name, unique_id, title=rel_type)

    nt = Network(
        height=height, 
        width="100%", 
        bgcolor="#ffffff", 
        font_color="#333", 
        notebook=False, 
        cdn_resources='remote'
    )
    nt.from_nx(G)
    
    # Physics for stability
    nt.set_options("""
    {
    "layout": {
        "randomSeed": 42
    },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)
    
    try:
        # Generate HTML and inject custom tooltip CSS
        html_content = nt.generate_html()
        
        # Inject custom tooltip styling into the HTML
        custom_tooltip_css = """
        <style>
        .vis-tooltip {
            background-color: #f8f9fa !important;
            color: #333 !important;
            border: 1px solid #9eb384 !important;
            border-radius: 6px !important;
            padding: 10px 14px !important;
            font-family: 'Consolas', 'Monaco', monospace !important;
            font-size: 12px !important;
            line-height: 1.5 !important;
            white-space: pre-line !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
            max-width: 300px !important;
        }
        </style>
        """
        
        # Insert the CSS right before </head>
        html_content = html_content.replace('</head>', custom_tooltip_css + '</head>')
        
        return html_content
    except Exception as e:
        print(f"Graph Gen Error: {e}")
        return ""