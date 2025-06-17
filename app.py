import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import requests
from typing import Dict, List
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import os
from streamlit_plotly_events import plotly_events
import re
from collections import Counter, defaultdict

# Set page config
st.set_page_config(
    page_title="Ayoda Capital Group - Struktur Perusahaan",
    page_icon="🏢",
    layout="wide"
)

# API Configuration
API_URL = "http://localhost:8000"  # Change this in production

# Title and description
st.title("Ayoda Capital Group - Visualisasi Struktur Perusahaan")
st.markdown("""
Dashboard ini menampilkan struktur kepemilikan dari Ayoda Capital Group dan anak perusahaannya.
""")

def fetch_companies() -> List[Dict]:
    """Fetch companies from the API"""
    response = requests.get(f"{API_URL}/companies/")
    return response.json()

def fetch_company_documents(company_id: int) -> List[Dict]:
    """Fetch documents for a specific company"""
    response = requests.get(f"{API_URL}/companies/{company_id}/documents")
    return response.json()

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
                # Increase ranksep for more vertical spacing
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB -Gnodesep=3.5 -Granksep=4.5')
            except Exception:
                pos = vertical_tree_layout(G, level_gap=4.0)  # More vertical spacing
        else:
            # Compute layout for each weakly connected component
            pos = {}
            for component in nx.weakly_connected_components(G):
                subgraph = G.subgraph(component)
                try:
                    sub_pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot', args='-Grankdir=TB -Gnodesep=3.0 -Granksep=3.0')
                except Exception:
                    try:
                        sub_pos = vertical_tree_layout(subgraph, level_gap=3.0)
                    except Exception:
                        sub_pos = nx.spring_layout(subgraph, k=10, iterations=300)
                pos.update(sub_pos)
    except Exception:
        pos = nx.spring_layout(G, k=10, iterations=300)
    # Assign default positions to any node missing from pos
    missing_nodes = set(G.nodes()) - set(pos.keys())
    if missing_nodes:
        print(f"Missing nodes in pos: {missing_nodes}")
        y_min = min((y for x, y in pos.values()), default=0)
        for i, node in enumerate(missing_nodes):
            pos[node] = (i * 10, y_min - 20)
    print(f"Nodes in G: {list(G.nodes())}")
    print(f"Nodes in pos: {list(pos.keys())}")
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
        if perc is not None:
            try:
                perc_int = int(round(float(perc)))
            except Exception:
                perc_int = perc
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_text.append(dict(x=mx, y=my, text=f"<b>{perc_int}%</b>", showarrow=False, font=dict(color='red', size=9), align='center'))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#B0B0B0'),
        hoverinfo='none',
        mode='lines'
    )
    # Use a single node_order for all node-related arrays
    node_order = list(G.nodes())
    node_x, node_y, node_text, node_colors, node_sizes, node_labels = [], [], [], [], [], []
    for node in node_order:
        if node not in pos:
            print(f"Warning: Node '{node}' not in pos, skipping.")
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        parent_str = ', '.join(parents) if parents else '-'
        child_str = ', '.join(children) if children else '-'
        name = G.nodes[node].get('name', node)
        hover_text = (
            f"<b style='color:white'>{name}</b><br>"
            f"<span style='color:white'>Induk: {parent_str}<br>Anak Perusahaan: {child_str}</span>"
        )
        node_text.append(hover_text)
        node_labels.append(node)
        if not parents:
            node_colors.append('#1976D2')
            node_sizes.append(50)
        else:
            node_colors.append('#43A047')
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
        )
    )
    node_trace.text = node_labels
    node_trace.hovertext = node_text
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={'text': 'Struktur Ayoda Capital Group', 'font': {'size': 28}},
            font=dict(color='white', size=18, family='Arial'),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=20, r=20, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#181825',
            paper_bgcolor='#181825',
            annotations=edge_text
        )
    )
    return fig, node_order, pos

def show_document_details(company_id: int):
    """Display document details for a selected company"""
    documents = fetch_company_documents(company_id)
    
    # st.subheader("Dokumen Perusahaan")
    for doc in documents:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{doc['document_type']}**")
            st.write(f"Nama file: {doc['name']}")
        with col2:
            if st.button("Lihat", key=f"view_{doc['id']}"):
                # In a real app, this would open the PDF viewer
                st.write(f"Opening document: {doc['name']}")

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

def create_visualization_subgraph(G):
    try:
        # Try dot layout first
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB -Gnodesep=3.5 -Granksep=4.5')
    except Exception as e:
        print(f"Dot layout failed: {e}")
        try:
            # Try vertical tree layout
            pos = vertical_tree_layout(G, level_gap=4.0)
        except Exception as e:
            print(f"Vertical tree layout failed: {e}")
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=10, iterations=300)
    
    # Ensure all nodes have positions
    missing_nodes = set(G.nodes()) - set(pos.keys())
    if missing_nodes:
        print(f"Missing nodes in pos: {missing_nodes}")
        y_min = min((y for x, y in pos.values()), default=0)
        for i, node in enumerate(missing_nodes):
            pos[node] = (i * 10, y_min - 20)
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        perc = G.edges[edge].get('percentage', None)
        if perc is not None:
            try:
                perc_int = int(round(float(perc)))
            except Exception:
                perc_int = perc
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_text.append(dict(x=mx, y=my, text=f"<b>{perc_int}%</b>", showarrow=False, font=dict(color='red', size=18), align='center'))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#B0B0B0'),
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
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        parent_str = ', '.join(parents) if parents else '-'
        child_str = ', '.join(children) if children else '-'
        name = G.nodes[node].get('name', node)
        hover_text = (
            f"<b style='color:white'>{name}</b><br>"
            f"<span style='color:white'>Induk: {parent_str}<br>Anak Perusahaan: {child_str}</span>"
        )
        node_text.append(hover_text)
        node_labels.append(node)
        if not parents:
            node_colors.append('#1976D2')
            node_sizes.append(50)
        else:
            node_colors.append('#43A047')
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
        )
    )
    node_trace.text = node_labels
    node_trace.hovertext = node_text
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            font=dict(color='white', size=18, family='Arial'),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#181825',
            paper_bgcolor='#181825',
            annotations=edge_text
        )
    )
    return fig, node_order, pos

def main():
    try:
        excel_path = 'corporate_structure_analysis_v2.xlsx'
        if os.path.exists(excel_path):
            G, _ = read_ownership_excel(excel_path)
            if G is None:
                return
        else:
            st.error(f"File '{excel_path}' not found.")
            return
        # Only show descendants of selected root (ACG or AGG)
        roots = [n for n in ['ACG', 'AGG'] if n in G.nodes]
        if not roots:
            st.error("No ACG or AGG node found in the graph.")
            return
        selected_root = st.selectbox("Pilih root perusahaan (ACG/AGG):", roots)
        # Get all descendants (subtree) of the selected root
        tree_nodes = nx.descendants(G, selected_root) | {selected_root}
        Gtree = G.subgraph(tree_nodes).copy()
        st.info(f"Menampilkan struktur dari: {selected_root}")
        fig, node_order, pos = create_visualization(Gtree, force_hierarchical=True)
        selected_points = plotly_events(fig, click_event=True, select_event=False, override_height=800, override_width='100%')
        node_clicked = None
        if selected_points:
            label = selected_points[0].get('customdata') or selected_points[0].get('text')
            if not label:
                # Fallback: try to match by x/y position if label is None
                x = selected_points[0].get('x')
                y = selected_points[0].get('y')
                for node, (x_pos, y_pos) in pos.items():
                    if abs(x_pos - x) < 1e-6 and abs(y_pos - y) < 1e-6:
                        label = node
                        break
            if label and label in Gtree.nodes:
                node_clicked = label
            else:
                st.warning(f"Clicked node label '{label}' not found in graph nodes.")
        show_node = node_clicked if node_clicked else None
        if show_node:
            try:
                details = {}
                # Try to get details from Gtree first
                if show_node in Gtree.nodes:
                    details = Gtree.nodes[show_node]
                # Then try G if not found
                elif show_node in G.nodes:
                    details = G.nodes[show_node]
                # Finally try subG if it exists
                elif 'subG' in locals() and show_node in subG.nodes:
                    details = subG.nodes[show_node]
                
                st.markdown("---")
                st.markdown(f"### Informasi {show_node}")
                # Use get() with fallback values for all attributes
                st.markdown(f"**Nama Lengkap:** {details.get('name', show_node)}")
                st.markdown(f"**Direktur Utama:** {details.get('direktur_utama', '-')}")
                st.markdown(f"**Direktur:** {details.get('direktur', '-')}")
                st.markdown(f"**Komisaris Utama:** {details.get('komisaris_utama', '-')}")
                st.markdown(f"**Komisaris:** {details.get('komisaris', '-')}")
                st.markdown(f"**Modal:** {details.get('modal', '-')}")
            except Exception as e:
                st.warning(f"Terjadi kesalahan saat menampilkan info node: {e}")
            # Show clear ownership table: 2 rows, owner and subsidiary
            direct_owners = []
            direct_subsidiaries = []
            for u, v, d in Gtree.edges(data=True):
                if v == show_node:
                    direct_owners.append((u, v, d))
                if u == show_node:
                    direct_subsidiaries.append((u, v, d))
            # Draw a second graph for direct ownership structure
            if direct_owners or direct_subsidiaries:
                subG = nx.DiGraph()
                # First, add all nodes with their attributes
                for u, v, d in direct_owners + direct_subsidiaries:
                    for node in [u, v]:
                        if node not in subG.nodes:
                            # Get attributes from Gtree, with fallback to G if not found
                            attrs = Gtree.nodes.get(node, {})
                            if not attrs and node in G.nodes:
                                attrs = G.nodes[node]
                            # Ensure name attribute exists
                            if 'name' not in attrs:
                                attrs['name'] = node
                            subG.add_node(node, **attrs)
                # Then add all edges
                for u, v, d in direct_owners + direct_subsidiaries:
                    subG.add_edge(u, v, **d)
                print(f"Subgraph nodes: {list(subG.nodes())}")
                print(f"Subgraph edges: {list(subG.edges())}")
                try:
                    fig2, _, _ = create_visualization_subgraph(subG)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Terjadi kesalahan saat menampilkan subgraph: {e}")
    except Exception as e:
        st.markdown(f'<div style="font-size:12px; color:#f88; margin-top:4em;">Terjadi kesalahan: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 