import streamlit as st
import pandas as pd
import numpy as np
import os
import pygwalker as pyg
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
import requests  # needed for LLM API calls
from pybliometrics.scopus import SerialTitle, ScopusSearch, init, create_config

# =============================================================================
# Pybliometrics Configuration
# =============================================================================
config_path = Path('./.config/pybliometrics.cfg')
if "SCOPUS_API_KEY" in st.secrets:
    api_key = st.secrets["SCOPUS_API_KEY"]
    os.environ["SCOPUS_API_KEY"] = api_key
    create_config(config_dir=config_path, keys=[api_key])
else:
    st.warning("No SCOPUS_API_KEY provided in secrets.toml. Check your configuration.")
init(config_path=config_path)

# =============================================================================
# LLM Integration – Convert a Query into a Scopus Query
# =============================================================================
def convert_query_with_llm(query, prompt_type="pubmed"):
    """
    Convert an input query into a valid Scopus query using an OpenAI LLM.
    
    Parameters:
    - query (str): The user provided query.
    - prompt_type (str): If "pubmed" then converts a PubMed query; otherwise, treats as a generic query.
    
    Returns:
    - (str) The converted Scopus query.
    """
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("No OPENAI_API_KEY provided in secrets.toml!")
        return None
    openai_api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    if prompt_type == "pubmed":
        prompt = f"Convert the following PubMed query into an equivalent Scopus query:\n\n{query}\n"
    else:
        prompt = f"Convert the following unformatted query into a valid Scopus query:\n\n{query}\n"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150,
    }
    response = requests.post(f"{openai_api_base}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"LLM API call failed: {response.text}")
        return None
    result = response.json()
    if result and "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    else:
        return None

# =============================================================================
# Global caching for SNIP lookups to help avoid duplicate API calls.
# =============================================================================
snip_cache = {}
def get_snip(journal_issn, pub_year):
    key = (journal_issn, pub_year)
    if key in snip_cache:
        return snip_cache[key]
    if pd.isna(journal_issn) or str(journal_issn).strip() == "" or pd.isna(pub_year):
        snip_cache[key] = np.nan
        return np.nan
    try:
        st_obj = SerialTitle(str(journal_issn), refresh=True, view='ENHANCED')
        if st_obj.sniplist and len(st_obj.sniplist) > 0:
            for yr, snip in st_obj.sniplist:
                if yr == pub_year:
                    snip_cache[key] = snip
                    return snip
            latest_snip = max(st_obj.sniplist, key=lambda x: x[0])[1]
            snip_cache[key] = latest_snip
            return latest_snip
        else:
            snip_cache[key] = np.nan
            return np.nan
    except Exception as e:
        st.error(f"Error retrieving SNIP for ISSN {journal_issn}: {e}")
        snip_cache[key] = np.nan
        return np.nan

# =============================================================================
# Data Loading and Processing Functions
# =============================================================================
@st.cache_data
def load_data(file):
    filename = file.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type! Please upload a CSV or Excel file.")
        df = pd.DataFrame()
    return df

@st.cache_data
def process_data(df):
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df['Year'] = df['publication_date'].dt.year
    df['Month'] = df['publication_date'].dt.month
    return df

def fetch_scopus_data(query):
    try:
        search = ScopusSearch(query)
        if not search.results:
            st.error("No results found for the query!")
            return pd.DataFrame()
        df = pd.DataFrame(search.results)
        if 'coverDate' in df.columns:
            df['publication_date'] = pd.to_datetime(df['coverDate'], errors='coerce')
        if 'publicationName' in df.columns:
            df['journal_name'] = df['publicationName']
        if 'issn' in df.columns:
            df['journal_issn'] = df['issn']
        return df
    except Exception as e:
        st.error("Error executing Scopus query: " + str(e))
        return pd.DataFrame()

@st.cache_data
def aggregate_counts(df):
    monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    yearly_counts = df.groupby('Year').size().reset_index(name='Count')
    return monthly_counts, yearly_counts

@st.cache_data
def enrich_with_snip(df):
    unique_pairs = df[['journal_issn', 'Year']].drop_duplicates()
    snip_mapping = {}
    for _, row in unique_pairs.iterrows():
        issn = row['journal_issn']
        year = row['Year']
        snip_mapping[(issn, year)] = get_snip(issn, year)
    df['SNIP'] = df.apply(lambda row: snip_mapping.get((row['journal_issn'], row['Year']), np.nan), axis=1)
    return df

# =============================================================================
# Function to Build the Co-Author Network 
# =============================================================================
def build_coauthor_network(df):
    G = nx.Graph()
    for authors in df['author_names']:
        # Split author names (assumed separated by ';') and strip whitespace
        author_list = [author.strip() for author in authors.split(';')]
        for pair in combinations(author_list, 2):
            if G.has_edge(*pair):
                G[pair[0]][pair[1]]['weight'] += 1
            else:
                G.add_edge(*pair, weight=1)
    return G

# =============================================================================
# Streamlit App Layout and Main Logic
# =============================================================================
st.title("Publication Metrics Dashboard")
st.markdown("""
This app allows you to explore publication metrics for the Division of Molecular and Translational BioMedicine.
You can either upload a spreadsheet (e.g., a publication report) **or** enter and execute a Scopus query.
The app aggregates publications over time, enriches them with SNIP 
(Source-Normalized Impact per Paper) values via the Elsevier API, and builds a co-author network.
""")

# ------------------------------------------------------------------------------
# Sidebar: LLM Query Conversion Options
# ------------------------------------------------------------------------------
with st.sidebar.expander("LLM Query Conversion", expanded=False):
    conversion_method = st.selectbox(
        "Select Conversion Method",
        ["None", "Convert PubMed Query to Scopus", "Convert Unformatted Query to Scopus"]
    )
    if conversion_method != "None":
        input_query = st.text_area("Enter your query", height=100)
        if st.button("Convert Query", key="llm_query_convert"):
            prompt_type = "pubmed" if conversion_method == "Convert PubMed Query to Scopus" else "generic"
            converted_query = convert_query_with_llm(input_query, prompt_type=prompt_type)
            if converted_query:
                st.text_area("Converted Scopus Query", value=converted_query, height=100)
            else:
                st.error("Conversion failed. Please check your input and API settings.")

# ------------------------------------------------------------------------------
# Sidebar: API & Data Input Settings
# ------------------------------------------------------------------------------
st.sidebar.header("API & Data Input Settings")
data_source = st.sidebar.radio("Select Data Source", ["Upload Spreadsheet", "Scopus Query"])
df = pd.DataFrame()

if data_source == "Upload Spreadsheet":
    uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = process_data(df)
elif data_source == "Scopus Query":
    st.sidebar.subheader("Enter Search Parameters")
    author = st.sidebar.text_input("Author")
    start_year = st.sidebar.text_input("Start Year")
    end_year = st.sidebar.text_input("End Year")
    afid = st.sidebar.text_input("Affiliation ID (AFID)")
    institution = st.sidebar.text_input("Affiliation Institution")
    scopus_query = st.sidebar.text_area("Or Enter Scopus Query Directly", height=100)
    
    def format_scopus_query(author, start_year, end_year, afid, institution):
        query = ""
        if author:
            query += f"AUTH({author}) AND "
        if start_year and end_year:
            query += f"PUBYEAR AFT {start_year} AND PUBYEAR BEF {end_year} AND "
        if afid:
            query += f"AF-ID({afid}) AND "
        if institution:
            institutions = [f"AFFIL({aff.strip()})" for aff in institution.split(",")]
            query += " AND ".join(institutions) + " AND "
        query = query.rstrip(" AND ")
        return query

    if st.sidebar.button("Execute Query"):
        if not scopus_query:
            scopus_query = format_scopus_query(author, start_year, end_year, afid, institution)
        with st.spinner("Executing Scopus query..."):
            df = fetch_scopus_data(scopus_query)
            if not df.empty:
                df = process_data(df)
                st.session_state.scopus_df = df

if "scopus_df" in st.session_state:
    df = st.session_state.scopus_df

# ------------------------------------------------------------------------------
# Main App: Display, Enrichment, and Visualization
# ------------------------------------------------------------------------------
if not df.empty:
    with st.spinner("Retrieving SNIP values from Elsevier..."):
        df = enrich_with_snip(df)
    # Reorder columns so key ones appear first
    desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
    other_columns = [col for col in df.columns if col not in desired_column_order]
    final_column_order = desired_column_order + other_columns
    df = df[final_column_order]
    
    st.subheader("Publications with Impact Factor (SNIP)")
    df = df.sort_values(by="SNIP", ascending=False)
    st.write(df)
    
    if st.button("Launch PyGWalker"):
        pyg.walk(df)  # Opens PyGWalker in a new browser window
    
    if "author_names" in df.columns:
        st.write("### Co-Author Network Visualization")
        coauthor_network = build_coauthor_network(df)

        def filter_network(graph, min_collaborations=1):
            filtered_graph = nx.Graph()
            for u, v, data in graph.edges(data=True):
                if data['weight'] >= min_collaborations:
                    if not filtered_graph.has_node(u):
                        filtered_graph.add_node(u)
                    if not filtered_graph.has_node(v):
                        filtered_graph.add_node(v)
                    filtered_graph.add_edge(u, v, weight=data['weight'])
            return filtered_graph

        min_collaborations = st.slider("Minimum Collaborations to Display", 1, 10, 3)
        filtered_coauthor_network = filter_network(coauthor_network, min_collaborations=min_collaborations)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(filtered_coauthor_network, seed=42, k=0.7)
        node_sizes = [100 + (degree * 10) for _, degree in filtered_coauthor_network.degree()]
        nx.draw_networkx_nodes(filtered_coauthor_network, pos, node_size=node_sizes, node_color="darkgreen")
        edge_widths = [filtered_coauthor_network[u][v]['weight'] for u, v in filtered_coauthor_network.edges()]
        nx.draw_networkx_edges(filtered_coauthor_network, pos, width=edge_widths, alpha=0.7, edge_color="gray")
        nx.draw_networkx_labels(filtered_coauthor_network, pos, font_size=8, font_color="black")
        plt.title("Filtered Co-Author Network", fontsize=14)
        plt.axis("off")
        st.pyplot(fig)
else:
    st.info("Please upload a publication file or execute a Scopus query from the sidebar.")

st.markdown("""
---
**Notes on the App and API Integration:**
• The “Upload Spreadsheet” option expects a file with at least the following columns:
  - publication_date
  - journal_name
  - journal_issn
• The “Scopus Query” option uses pybliometrics’ ScopusSearch to retrieve publication data.
• SNIP values are retrieved via pybliometrics’ SerialTitle API (view: "ENHANCED").
• Duplicate lookups for the same journal and year are avoided.
• LLM Query Conversion requires your OpenAI credentials (refer to OPENAI_API_KEY and OPENAI_API_BASE in secrets.toml).
• Elsevier API access requires your SCOPUS_API_KEY (available from the Elsevier Developer Portal).
""")