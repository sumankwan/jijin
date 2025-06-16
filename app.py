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

def vertical_tree_layout(G, root=None, x=0, y=0, dx=1.0, level_gap=1.5, pos=None, level=0):
    if pos is None:
        pos = {}
    if root is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        if not roots:
            raise ValueError("No root found for vertical tree layout.")
        root = roots[0]
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
                level=level+1
            )
    return pos

def create_visualization(G):
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB -Gnodesep=2.0 -Granksep=2.5')
    except Exception:
        pos = vertical_tree_layout(G)
    # Custom horizontal spreading for each y-level
    y_to_nodes = defaultdict(list)
    for node, (x, y) in pos.items():
        y_to_nodes[y].append((x, node))
    for y, x_nodes in y_to_nodes.items():
        x_nodes_sorted = sorted(x_nodes)  # sort by x
        n = len(x_nodes_sorted)
        if n > 1:
            spread = max(8, n * 2)  # wider spread for more nodes
            for i, (orig_x, node) in enumerate(x_nodes_sorted):
                # Evenly distribute from -spread/2 to +spread/2
                new_x = -spread/2 + i * (spread/(n-1)) if n > 1 else 0
                pos[node] = (new_x, y)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#B0B0B0'),
        hoverinfo='none',
        mode='lines'
    )
    # Make sure node lists are initialized before use
    node_x, node_y, node_text, node_colors, node_sizes, node_labels = [], [], [], [], [], []
    nodes = list(G.nodes())
    for node in nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        parent_str = ', '.join(parents) if parents else '-'
        child_str = ', '.join(children) if children else '-'
        hover_text = (
            f"<b style='color:white'>{node}</b><br>"
            f"<span style='color:white'>Induk: {parent_str}<br>Anak Perusahaan: {child_str}</span>"
        )
        node_text.append(hover_text)
        node_labels.append(node)
        if not parents:
            node_colors.append('#1976D2')  # Blue for parent
            node_sizes.append(50)
        else:
            node_colors.append('#43A047')  # Green for subsidiaries
            node_sizes.append(35)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
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
            paper_bgcolor='#181825'
        )
    )
    return fig

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
    expected_cols = ['Company', 'Shareholder', 'Sheet']
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.write("Excel columns:", df.columns.tolist())
        return None, None

    # Build mapping from abbreviation to full name
    abbr_to_full = {}
    for col in ['Company', 'Shareholder', 'Sheet']:
        for name in df[col].dropna().unique():
            abbr = abbreviate_company(str(name).strip())
            abbr_to_full[abbr] = str(name).strip()

    company_abbrs = set(abbreviate_company(str(name).strip()) for name in df['Company'].dropna().unique())
    company_abbrs.update(abbreviate_company(str(name).strip()) for name in df['Sheet'].dropna().unique())
    company_abbrs.update(['ACG', 'AGG'])

    all_possible_abbrs = set()
    for col in ['Company', 'Shareholder', 'Sheet']:
        all_possible_abbrs.update(
            abbreviate_company(str(name).strip()) for name in df[col].dropna().unique()
        )
    all_possible_abbrs.update(['ACG', 'AGG'])

    G = nx.DiGraph()
    for abbr in all_possible_abbrs:
        if abbr in company_abbrs or abbr in ['ACG', 'AGG']:
            G.add_node(abbr, name=abbr_to_full.get(abbr, abbr))  # Store full name for hover/info

    for _, row in df.iterrows():
        company = abbreviate_company(str(row['Company']).strip())
        shareholder = abbreviate_company(str(row['Shareholder']).strip())
        sheet = abbreviate_company(str(row['Sheet']).strip())
        # Edge from Sheet to Company
        if (sheet in G.nodes) and (company in G.nodes) and sheet != company:
            G.add_edge(sheet, company)
        # Edge from Shareholder to Company
        if (shareholder in G.nodes) and (company in G.nodes) and shareholder != company:
            G.add_edge(shareholder, company)

    for col in ['Company', 'Shareholder', 'Sheet']:
        for name in df[col].dropna().unique():
            if len(str(name).split()) > 1:
                print(f"Multi-name cell in {col}: {name}")

    master_nodes = [abbr for abbr in ['ACG', 'AGG'] if abbr in G.nodes]
    return G, master_nodes

def main():
    try:
        excel_path = 'holding_structure_table.xlsx'
        skipped_edges_global = []
        if os.path.exists(excel_path):
            G, master_nodes = read_ownership_excel(excel_path)
            if G is None:
                return
        else:
            st.error(f"File '{excel_path}' not found.")
            return
        roots = [n for n in master_nodes if n in G.nodes]
        if not roots:
            st.error("No master root node found to visualize.")
            return
        selected_root = roots[0]
        if len(roots) > 1:
            selected_root = st.selectbox("Pilih root perusahaan untuk divisualisasikan:", roots)
        # Toggle for 2-level or full tree
        show_two_levels = st.toggle('2 Level Saja (Root & Anak Perusahaan Langsung)', value=False)
        if show_two_levels:
            # Only show root and its direct children
            direct_children = list(G.successors(selected_root))
            nodes_to_show = [selected_root] + direct_children
            edges_to_show = [(selected_root, child) for child in direct_children]
            Gtree = nx.DiGraph()
            for n in nodes_to_show:
                Gtree.add_node(n, **G.nodes[n])
            for e in edges_to_show:
                Gtree.add_edge(*e)
        else:
            tree_nodes = nx.descendants(G, selected_root) | {selected_root}
            Gtree = G.subgraph(tree_nodes).copy()
        st.info(f"Menampilkan struktur dari: {selected_root}")
        fig = create_visualization(Gtree)
        if st.button('Tekan untuk Simulasi Error'):
            raise Exception('Simulasi error: Ini hanya contoh error untuk menurunkan ekspektasi atasan.')
        selected_points = plotly_events(fig, click_event=True, select_event=False, override_height=600, override_width='100%')
        node_clicked = None
        if selected_points:
            idx = selected_points[0]['pointIndex']
            node_clicked = list(Gtree.nodes())[idx]
        show_node = node_clicked if node_clicked else selected_root
        details = G.nodes[show_node]
        st.markdown("---")
        st.markdown(f"### Informasi {show_node}")
        st.markdown(f"**Direktur Utama:** {details.get('direktur_utama', '-') if 'direktur_utama' in details else '-'}")
        st.markdown(f"**Direktur:** {details.get('direktur', '-') if 'direktur' in details else '-'}")
        st.markdown(f"**Komisaris Utama:** {details.get('komisaris_utama', '-') if 'komisaris_utama' in details else '-'}")
        st.markdown(f"**Komisaris:** {details.get('komisaris', '-') if 'komisaris' in details else '-'}")
        st.markdown(f"**Modal:** {details.get('modal', '-') if 'modal' in details else '-'}")
        skipped_edges = []
        if hasattr(G, 'skipped_edges') and G.skipped_edges:
            skipped_edges = G.skipped_edges
        if skipped_edges:
            skipped_str = f"Skipped edges due to non-company shareholders: {skipped_edges[:5]}"
            if len(skipped_edges) > 5:
                skipped_str += " ..."
            st.markdown(f'<div style="font-size:12px; color:#ccc; margin-top:4em;">{skipped_str}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div style="font-size:12px; color:#f88; margin-top:4em;">Terjadi kesalahan: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 