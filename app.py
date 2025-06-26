import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import os
from streamlit_plotly_events import plotly_events
import re
from collections import Counter, defaultdict
import numpy as np

# Set page config
st.set_page_config(
    page_title="Ayoda Capital Group - Struktur Perusahaan",
    page_icon="🏢",
    layout="wide"
)

# Title and description
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Ayoda Capital Group - Struktur Organisasi</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            min-width: 140px;
            max-width: 180px;
        }
        @media (max-width: 768px) {
            /* Mobile-specific styles */
            .stPlotlyChart {
                height: 400px !important;
            }
            .stButton > button {
                width: 100% !important;
                height: 40px !important;
                font-size: 16px !important;
            }
            .stSelectbox > div > div {
                font-size: 16px !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Removed API functions - app now works completely locally with Excel data

def create_network_graph(companies: List[Dict]):
    G = nx.DiGraph()
    for company in companies:
        G.add_node(company['id'], 
                  name=company['name'],
                  ownership=company.get('ownership_percentage', None),
                  direktur=company.get('direktur', ''),
                  komisaris=company.get('komisaris', ''),
                  bod=company.get('bod', ''),
                  boc=company.get('boc', ''),
                  modal=company.get('modal', None))
        if company.get('parent_id'):
            G.add_edge(company['parent_id'], company['id'])
    return G

def vertical_tree_layout(G, root=None, x=0, y=0, dx=1.0, level_gap=1.5, pos=None, level=0, visited=None):
    if pos is None:
        pos = {}
    if visited is None:
        visited = set()
    if root is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        if not roots:
            raise ValueError("No root found for vertical tree layout.")
        root = roots[0]
    if root in visited:
        return pos
    visited.add(root)
    pos[root] = (x, y)
    children = list(G.successors(root))
    if children:
        width = dx * (len(children) - 1)
        for i, child in enumerate(children):
            pos = vertical_tree_layout(
                G, child,
                x - width/2.0 + i*dx if len(children) > 1 else x,
                y - level_gap,
                dx=dx/1.5,
                level_gap=level_gap,
                pos=pos,
                level=level+1,
                visited=visited
            )
    return pos

def create_visualization(G, force_hierarchical=False):
    try:
        if force_hierarchical:
            try:
                pos = vertical_tree_layout(G, level_gap=8.0)
            except Exception as e:
                print(f"Vertical tree layout failed: {e}")
                pos = nx.spring_layout(G, k=20, iterations=300)
        else:
            pos = {}
            for component in nx.weakly_connected_components(G):
                subgraph = G.subgraph(component)
                try:
                    sub_pos = vertical_tree_layout(subgraph, level_gap=8.0)
                except Exception:
                    sub_pos = nx.spring_layout(subgraph, k=20, iterations=300)
                pos.update(sub_pos)
    except Exception as e:
        print(f"Layout failed: {e}")
        pos = nx.spring_layout(G, k=20, iterations=300)

    missing_nodes = set(G.nodes()) - set(pos.keys())
    if missing_nodes:
        print(f"Missing nodes in pos: {missing_nodes}")
        y_min = min((y for x, y in pos.values()), default=0)
        for i, node in enumerate(missing_nodes):
            pos[node] = (i * 10, y_min - 20)
    y_to_nodes = defaultdict(list)
    for node, (x, y) in pos.items():
        y_to_nodes[y].append((x, node))
    for y, x_nodes in y_to_nodes.items():
        x_nodes_sorted = sorted(x_nodes)
        n = len(x_nodes_sorted)
        if n > 1:
            spread = max(8, n * 2)
            for i, (orig_x, node) in enumerate(x_nodes_sorted):
                new_x = -spread/2 + i * (spread/(n-1)) if n > 1 else 0
                pos[node] = (new_x, y)
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        perc = G.edges[edge].get('percentage', None)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if perc is not None and str(perc).strip() != "":
            try:
                perc_int = int(round(float(perc)))
                label = f"<b>{perc_int}%</b>"
            except Exception:
                label = f"<b>{perc}</b>"
        else:
            label = "<b>?</b>"  # Or use '-' or 'N/A' if you prefer
        if not any(abs(mx - ann['x']) < 1e-6 and abs(my - ann['y']) < 1e-6 for ann in edge_text):
            edge_text.append(dict(x=mx, y=my, text=label, showarrow=False, font=dict(color='red', size=6), align='center'))  # Reduced to size=6 for master graph
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#B0B0B0'),
        hoverinfo='none',
        mode='lines'
    )
    # Use a single node_order for all node-related arrays
    node_order = list(G.nodes())
    node_x, node_y, node_text, node_colors, node_sizes, node_labels = [], [], [], [], [], []
    for node in node_order:
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Abbreviation for label
        name = G.nodes[node].get('name', node)
        abbr = abbreviate_company(name)
        node_labels.append(abbr)
        # Full name for hover
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        parent_str = ', '.join([G.nodes[p].get('name', p) for p in parents]) if parents else '-'
        child_str = ', '.join([G.nodes[c].get('name', c) for c in children]) if children else '-'
        hover_text = (
            f"<b style='color:white'>{name}</b><br>"
            f"<span style='color:white'>Induk: {parent_str}<br>Anak Perusahaan: {child_str}</span>"
        )
        node_text.append(hover_text)
        node_colors.append('#1976D2')  # Blue for all nodes
        node_sizes.append(40)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        customdata=node_labels,  # This is crucial for click events
        textfont=dict(color='white', size=12, family='Arial'),
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=6, color='#FFFFFF'),
            opacity=0.95,
            symbol='circle'
        ),
        hovertext=node_text,  # This ensures the hover text is used
        hovertemplate='%{hovertext}<extra></extra>'  # This helps with click detection
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            # title={'text': 'Struktur Ayoda Capital Group', 'font': {'size': 28}},
            font=dict(color='white', size=18, family='Arial'),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#181825',
            paper_bgcolor='#181825',
            annotations=edge_text,
            height=1200,  # Increased height
            autosize=True
        )
    )

    # After creating edge_trace, add arrows for each edge
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Draw an arrow from parent (shareholder) to child (subsidiary)
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=4,        # Medium, classic arrowhead
            arrowsize=1.0,      # Reduced from 1.5 to 1.0
            arrowwidth=1.5,     # Reduced from 2.5 to 1.5
            arrowcolor='#1976D2',  # Blue color
            opacity=1.0,        # Fully opaque
            standoff=8          # Reduced from 10 to 8
        )
    return fig, node_order, pos

# Document functionality removed for local deployment

def abbreviate(name):
    name = name.strip()
    if name.upper().startswith('PT '):
        name = name[3:].strip()
    words = name.split()
    return ''.join([c for word in words for c in word if c.isupper()])

def abbreviate_company(name):
    # Remove leading "PT" (case-insensitive)
    name = re.sub(r'^PT\s+', '', name, flags=re.IGNORECASE)
    # Take all uppercase letters from the remaining words
    abbreviation = ''.join(word[0] for word in name.split() if word[0].isupper())
    return abbreviation

def read_ownership_excel(filepath):
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    expected_cols = ['Subsidiary', 'Shareholder', 'Ownership_Percentage']
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.write("Excel columns:", df.columns.tolist())
        return None, None

    # Build mapping from abbreviation to full name
    abbr_to_full = {}
    all_abbrs = set()
    for col in ['Subsidiary', 'Shareholder']:
        for name in df[col].dropna().unique():
            abbr = abbreviate_company(str(name).strip())
            if abbr:
                abbr_to_full[abbr] = str(name).strip()
                all_abbrs.add(abbr)

    G = nx.DiGraph()
    for abbr in all_abbrs:
        G.add_node(abbr, name=abbr_to_full.get(abbr, abbr))

    for _, row in df.iterrows():
        sub = abbreviate_company(str(row['Subsidiary']).strip())
        shr = abbreviate_company(str(row['Shareholder']).strip())
        perc = row['Ownership_Percentage']
        if not sub or not shr:
            print(f"Skipping edge with empty abbreviation: Shareholder='{row['Shareholder']}', Subsidiary='{row['Subsidiary']}'")
            continue
        if (shr in G.nodes) and (sub in G.nodes) and shr != sub:
            G.add_edge(shr, sub, percentage=perc)
        else:
            if shr not in G.nodes or sub not in G.nodes:
                print(f"Warning: Edge references missing node(s): Shareholder='{shr}', Subsidiary='{sub}'")

    master_nodes = [n for n, d in G.in_degree() if d == 0]
    print(f"All node abbreviations: {list(G.nodes())}")
    print(f"All edges: {list(G.edges())}")
    return G, master_nodes

def get_subgraph_for_company(G, selected_company):
    # Get all parents of the selected company
    parents = set()
    for parent in G.predecessors(selected_company):
        parents.add(parent)
    
    # Get all subsidiaries of those parents (siblings + selected company)
    all_siblings = set()
    for parent in parents:
        for subsidiary in G.successors(parent):
            all_siblings.add(subsidiary)
    
    # Get ONLY DIRECT subsidiaries of the selected company (no grandchildren)
    direct_subsidiaries = set()
    for subsidiary in G.successors(selected_company):
        direct_subsidiaries.add(subsidiary)
    
    # Combine all nodes
    all_nodes = parents.union(all_siblings).union(direct_subsidiaries)
    
    # If no parents (root node like ACG), just show the node and its direct subsidiaries
    if not parents:
        all_nodes = {selected_company}.union(direct_subsidiaries)
    
    return G.subgraph(all_nodes)

def create_visualization_subgraph(G, selected_node, pos=None):
    subG = G.copy()
    if pos is None:
        try:
            pos = vertical_subgraph_layout(subG, selected_node)
        except Exception as e:
            print(f"Vertical subgraph layout failed: {e}")
            pos = nx.spring_layout(subG, k=20, iterations=300)

    # Ensure all nodes have positions
    missing_nodes = set(subG.nodes()) - set(pos.keys())
    if missing_nodes:
        y_min = min((y for x, y in pos.values()), default=0)
        for i, node in enumerate(missing_nodes):
            pos[node] = (i * 10, y_min - 20)

    edge_x, edge_y, edge_text = [], [], []
    seen_edges = set()
    for edge in subG.edges():
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        perc = subG.edges[edge].get('percentage', None)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if perc is not None and str(perc).strip() != "":
            try:
                perc_int = int(round(float(perc)))
                label = f"<b>{perc_int}%</b>"
            except Exception:
                label = f"<b>{perc}</b>"
        else:
            label = "<b>?</b>"  # Or use '-' or 'N/A' if you prefer
        # Only add annotation if not already present at this location
        if not any(abs(mx - ann['x']) < 1e-6 and abs(my - ann['y']) < 1e-6 for ann in edge_text):
            edge_text.append(dict(x=mx, y=my, text=label, showarrow=False, font=dict(color='red', size=8), align='center'))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#B0B0B0'),
        hoverinfo='none',
        mode='lines'
    )

    node_order = list(subG.nodes())
    node_x, node_y, node_text, node_colors, node_sizes, node_labels, node_hovertext = [], [], [], [], [], [], []
    for node in node_order:
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        name = subG.nodes[node].get('name', node)
        abbr = abbreviate_company(name)
        node_labels.append(abbr)
        node_hovertext.append(name)
        
        # Restore the color coding
        if node == selected_node:
            node_colors.append('#FFC107')  # Yellow for selected
            node_sizes.append(50)
        elif selected_node in subG.nodes and node in list(subG.predecessors(selected_node)):
            node_colors.append('#1976D2')  # Blue for parent
            node_sizes.append(40)
        elif selected_node in subG.nodes and node in list(subG.successors(selected_node)):
            node_colors.append('#43A047')  # Green for subsidiary
            node_sizes.append(40)
        else:
            node_colors.append('#FFFFFF')  # White for other nodes (covering them)
            node_sizes.append(35)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        customdata=node_labels,
        textfont=dict(color='white', size=12, family='Arial'),
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=6, color='#FFFFFF'),
            opacity=0.95,
            symbol='circle'
        ),
        hovertext=node_hovertext,  # <-- This ensures full name on hover
        hovertemplate='%{hovertext}<extra></extra>'  # This helps with click detection
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            font=dict(color='white', size=18, family='Arial'),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#181825',
            paper_bgcolor='#181825',
            annotations=edge_text,
            autosize=True
        )
    )

    # Add arrows for directionality
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=4,        # Medium, classic arrowhead
            arrowsize=1.0,      # Reduced from 1.5 to 1.0
            arrowwidth=1.5,     # Reduced from 2.5 to 1.5
            arrowcolor='#1976D2',  # Blue color
            opacity=1.0,        # Fully opaque
            standoff=8          # Reduced from 10 to 8
        )
    return fig, node_order, pos

def vertical_subgraph_layout(G, selected_node):
    pos = {}

    # Parents at the top (y=2)
    parents = list(G.predecessors(selected_node))
    n_parents = len(parents)
    for i, parent in enumerate(parents):
        pos[parent] = (i - (n_parents-1)/2, 2)

    # Siblings and selected node at the middle level (y=1)
    siblings = set()
    for parent in parents:
        siblings.update(G.successors(parent))
    siblings.discard(selected_node)

    # --- Fixed positioning system ---
    all_middle_nodes = [selected_node] + list(siblings)
    
    # Create a fixed order based on alphabetical sorting of full names
    # This ensures the same positions regardless of which node is selected
    all_middle_nodes.sort(key=lambda x: G.nodes[x].get('name', x))
    
    # Position all nodes in their fixed alphabetical order
    n_middle = len(all_middle_nodes)
    for i, node in enumerate(all_middle_nodes):
        pos[node] = (i - (n_middle-1)/2, 1)

    # Direct subsidiaries at the bottom (y=0)
    subsidiaries = set(G.successors(selected_node))
    bottom_nodes = list(subsidiaries)
    bottom_nodes.sort()  # Sort for consistency
    n_bottom = len(bottom_nodes)
    for i, node in enumerate(bottom_nodes):
        pos[node] = (i - (n_bottom-1)/2, 0)

    return pos

def create_highlighted_master_graph(G, selected_node, pos=None):
    """Create master graph with highlighted subgraph nodes"""
    if pos is None:
        try:
            # Use a much more spread out layout
            pos = vertical_tree_layout(G, level_gap=20.0)  # Increased from 12.0 to 20.0
        except Exception as e:
            print(f"Vertical tree layout failed: {e}")
            pos = nx.spring_layout(G, k=50, iterations=1000)  # Increased k and iterations for much better spacing

    # Get subgraph nodes to highlight
    subG = get_subgraph_for_company(G, selected_node)
    subgraph_nodes = set(subG.nodes())

    # Improve node spacing by spreading out nodes at the same level
    y_to_nodes = defaultdict(list)
    for node, (x, y) in pos.items():
        y_to_nodes[y].append((x, node))
    
    # Spread out nodes at each level with much more spacing
    for y, x_nodes in y_to_nodes.items():
        x_nodes_sorted = sorted(x_nodes)
        n = len(x_nodes_sorted)
        if n > 1:
            spread = max(30, n * 8)  # Increased spread significantly for better separation
            for i, (orig_x, node) in enumerate(x_nodes_sorted):
                new_x = -spread/2 + i * (spread/(n-1)) if n > 1 else 0
                pos[node] = (new_x, y)

    # Create the visualization with highlighted nodes
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        perc = G.edges[edge].get('percentage', None)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if perc is not None and str(perc).strip() != "":
            try:
                perc_int = int(round(float(perc)))
                label = f"<b>{perc_int}%</b>"
            except Exception:
                label = f"<b>{perc}</b>"
        else:
            label = "<b>?</b>"
        if not any(abs(mx - ann['x']) < 1e-6 and abs(my - ann['y']) < 1e-6 for ann in edge_text):
            edge_text.append(dict(x=mx, y=my, text=label, showarrow=False, font=dict(color='red', size=6), align='center'))  # Reduced to size=6 for highlighted master graph

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#B0B0B0'),
        hoverinfo='none',
        mode='lines'
    )

    node_order = list(G.nodes())
    node_x, node_y, node_text, node_colors, node_sizes, node_labels = [], [], [], [], [], []
    for node in node_order:
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        name = G.nodes[node].get('name', node)
        abbr = abbreviate_company(name)
        node_labels.append(abbr)
        
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        parent_str = ', '.join([G.nodes[p].get('name', p) for p in parents]) if parents else '-'
        child_str = ', '.join([G.nodes[c].get('name', c) for c in children]) if children else '-'
        hover_text = (
            f"<b style='color:white'>{name}</b><br>"
            f"<span style='color:white'>Induk: {parent_str}<br>Anak Perusahaan: {child_str}</span>"
        )
        node_text.append(hover_text)
        
        # Color coding: highlight subgraph nodes
        if node == selected_node:
            node_colors.append('#FFC107')  # Yellow for selected
            node_sizes.append(45)  # Slightly smaller
        elif node in subgraph_nodes:
            node_colors.append('#FF5722')  # Orange for subgraph nodes
            node_sizes.append(40)  # Slightly smaller
        else:
            node_colors.append('#FFFFFF')  # White for other nodes (covering them)
            node_sizes.append(30)  # Smaller for background nodes

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        customdata=node_labels,
        textfont=dict(color='white', size=10, family='Arial'),  # Smaller font
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=4, color='#FFFFFF'),  # Thinner border
            opacity=0.95,
            symbol='circle'
        ),
        hovertext=node_text,
        hovertemplate='%{hovertext}<extra></extra>'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            font=dict(color='white', size=16, family='Arial'),  # Smaller font
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=20, r=20, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#181825',
            paper_bgcolor='#181825',
            annotations=edge_text,
            height=1600  # Increased height for much better spacing
        )
    )

    # Add arrows
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=4,
            arrowsize=0.8,  # Smaller arrows
            arrowwidth=1.2,  # Thinner arrows
            arrowcolor='#1976D2',
            opacity=1.0,
            standoff=6  # Smaller standoff
        )
    return fig, node_order, pos

def main():
    import datetime
    print(f"MAIN RUN at {datetime.datetime.now()}")
    print("Session state at start:", dict(st.session_state))
    try:
        # Initialize session state for view mode
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "Pilih Perusahaan"
        if 'show_highlighted_master' not in st.session_state:
            st.session_state.show_highlighted_master = False
        if 'highlighted_node' not in st.session_state:
            st.session_state.highlighted_node = None
        
        # Sidebar toggle
        view_mode = st.sidebar.radio(
            "Tampilan",
            ("Pilih Perusahaan", "Master Graph"),
            index=0 if st.session_state.view_mode == "Pilih Perusahaan" else 1
        )
        
        # Update session state when view mode changes
        if view_mode != st.session_state.view_mode:
            st.session_state.view_mode = view_mode
            # Reset selected node to ACG when switching to Pilih Perusahaan
            if view_mode == "Pilih Perusahaan":
                st.session_state.selected_node = "ACG"
            st.rerun()
            st.stop()

        # Build graph for selected root (ACG by default)
        if 'selected_root' not in st.session_state:
            st.session_state.selected_root = "ACG"
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = "ACG"

        excel_path = 'corporate_structure_analysis_v2.xlsx'
        if os.path.exists(excel_path):
            G, _ = read_ownership_excel(excel_path)
            if G is None:
                st.stop()
        else:
            st.error(f"File '{excel_path}' not found.")
            st.stop()

        tree_nodes = nx.descendants(G, st.session_state.selected_root) | {st.session_state.selected_root}
        Gtree = G.subgraph(tree_nodes).copy()
        abbr_to_full = {node: Gtree.nodes[node].get('name', node) for node in Gtree.nodes}
        full_to_abbr = {v: k for k, v in abbr_to_full.items()}
        all_full_names = [abbr_to_full[node] for node in sorted(Gtree.nodes())]

        if st.session_state.view_mode == "Pilih Perusahaan":
            # Company dropdown
            selected_full_name = abbr_to_full[st.session_state.selected_node]
            new_full_name = st.selectbox(
                "Pilih Perusahaan",
                all_full_names,
                index=all_full_names.index(selected_full_name)
            )
            if new_full_name != selected_full_name:
                st.session_state.selected_node = full_to_abbr[new_full_name]
                st.rerun()
                st.stop()
            selected_node = full_to_abbr[selected_full_name]

            # Info card
            details = Gtree.nodes[selected_node]
            st.markdown(f"""
            <div style='background:#232339;padding:1.5em 1em 1em 1em;border-radius:12px;margin-bottom:1em;box-shadow:0 2px 8px #0002;'>
                <b>Nama:</b> {details.get('name', selected_node)}<br>
                <b>Direktur:</b> {details.get('direktur', '-') }<br>
                <b>Komisaris:</b> {details.get('komisaris', '-') }<br>
                <b>Modal:</b> {details.get('modal', '-') }
            </div>
            """, unsafe_allow_html=True)

            # Legend
            st.markdown("""
            <div style='background:#232339;padding:1em;border-radius:8px;margin-bottom:1em;box-shadow:0 2px 8px #0002;'>
                <div style='display:flex;justify-content:center;gap:2em;flex-wrap:wrap;'>
                    <div style='display:flex;align-items:center;gap:0.5em;'>
                        <div style='width:20px;height:20px;background:#FFC107;border-radius:50%;'></div>
                        <span>Perusahaan Terpilih</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:0.5em;'>
                        <div style='width:20px;height:20px;background:#1976D2;border-radius:50%;'></div>
                        <span>Perusahaan Induk</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:0.5em;'>
                        <div style='width:20px;height:20px;background:#43A047;border-radius:50%;'></div>
                        <span>Anak Perusahaan</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:0.5em;'>
                        <div style='width:20px;height:20px;background:#B0B0B0;border-radius:50%;'></div>
                        <span>Perusahaan Saudara</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Subgraph with vertical, centered layout
            subG = get_subgraph_for_company(Gtree, selected_node)
            try:
                pos = vertical_subgraph_layout(subG, selected_node)
            except Exception as e:
                print(f"Vertical subgraph layout failed: {e}")
                pos = nx.spring_layout(subG, k=20, iterations=300)
            fig2, node_order2, _ = create_visualization_subgraph(subG, selected_node, pos=pos)
            
            # Add click callback to the figure
            fig2.update_layout(
                clickmode='event+select',
                dragmode=False
            )
            
            # Use plotly_events for node clicks
            selected_points2 = plotly_events(
                fig2, 
                click_event=True, 
                select_event=False, 
                override_height=600, 
                override_width='100%', 
                key=f"subgraph_{selected_node}"
            )
            
            # Button below subgraph
            if st.button("Lihat di Master Graph", key=f"show_master_{selected_node}"):
                st.session_state.show_highlighted_master = True
                st.session_state.highlighted_node = selected_node
                st.rerun()
                st.stop()
            
            if selected_points2:
                print(f"Subgraph clicked: {selected_points2}")
                point_data = selected_points2[0]
                print(f"Point data keys: {point_data.keys()}")
                print(f"Full point data: {point_data}")
                
                # Try different ways to get the label
                label2 = None
                if 'customdata' in point_data and point_data['customdata']:
                    label2 = point_data['customdata']
                elif 'text' in point_data and point_data['text']:
                    label2 = point_data['text']
                elif 'pointIndex' in point_data:
                    idx = point_data['pointIndex']
                    if idx < len(node_order2):
                        label2 = node_order2[idx]
                
                print(f"Extracted label: {label2}")
                print(f"Available nodes: {list(Gtree.nodes())}")
                
                if label2 and label2 in Gtree.nodes:
                    print(f"Setting selected node to: {label2}")
                    st.session_state.selected_node = label2
                    st.rerun()
                    st.stop()
                else:
                    print(f"Label {label2} not found in nodes")

            # Display highlighted master graph if requested
            if st.session_state.get('show_highlighted_master', False) and st.session_state.get('highlighted_node') == selected_node:
                st.markdown("<h4 style='text-align:center;'>Posisi dalam Master Graph</h4>", unsafe_allow_html=True)
                highlighted_fig, _, _ = create_highlighted_master_graph(Gtree, selected_node)
                st.plotly_chart(highlighted_fig, use_container_width=True, height=1200, key="highlighted_master")
                
                if st.button("Kembali ke Subgraph", key="back_to_subgraph"):
                    st.session_state.show_highlighted_master = False
                    st.rerun()
                    st.stop()

        elif st.session_state.view_mode == "Master Graph":
            st.markdown("<h4 style='text-align:center;'>Struktur Lengkap (Master Graph)</h4>", unsafe_allow_html=True)
            fig, node_order, pos = create_visualization(Gtree, force_hierarchical=True)
            
            # Add click callback to the figure
            fig.update_layout(
                clickmode='event+select',
                dragmode=False
            )
            
            # Use plotly_events for node clicks
            selected_points = plotly_events(
                fig, 
                click_event=True, 
                select_event=False, 
                override_height=1200, 
                override_width='100%', 
                key="master_graph"
            )
            
            if selected_points:
                print(f"Master graph clicked: {selected_points}")
                point_data = selected_points[0]
                print(f"Point data keys: {point_data.keys()}")
                print(f"Full point data: {point_data}")
                
                # Try different ways to get the label
                label = None
                if 'customdata' in point_data and point_data['customdata']:
                    label = point_data['customdata']
                elif 'text' in point_data and point_data['text']:
                    label = point_data['text']
                elif 'pointIndex' in point_data:
                    idx = point_data['pointIndex']
                    if idx < len(node_order):
                        label = node_order[idx]
                
                print(f"Extracted label: {label}")
                print(f"Available nodes: {list(Gtree.nodes())}")
                
                if label and label in Gtree.nodes:
                    print(f"Switching to Pilih Perusahaan with node: {label}")
                    st.session_state.view_mode = "Pilih Perusahaan"
                    st.session_state.selected_node = label
                    st.rerun()
                    st.stop()
                else:
                    print(f"Label {label} not found in nodes")
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, height=1200, key="master_graph_chart")

    except Exception as e:
        st.markdown(f'<div style="font-size:12px; color:#f88; margin-top:4em;">Terjadi kesalahan: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()