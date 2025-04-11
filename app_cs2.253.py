import streamlit as st
import pandas as pd
import numpy as np
import os
import pygwalker as pyg
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations
from pathlib import Path
import datetime
from io import BytesIO
import re
from docx import Document
import requests
import json
import time
from urllib.parse import quote
import requests
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
    - query (str): The user-provided query.
    - prompt_type (str): Specifies the type of query, e.g., "pubmed" or generic.

    Returns:
    - (str) The converted Scopus query.
    """
    def preprocess_date_range(input_query):
        """
        Detect and process date ranges within the input query.
        Returns a reformatted string with consistent PUBYEAR conditions.
        """
        import re
        date_range_pattern = r"(\d{4})-(\d{4})"
        match = re.search(date_range_pattern, input_query)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            lower_bound = start_year - 1
            upper_bound = end_year + 1
            date_query = f"PUBYEAR > {lower_bound} AND PUBYEAR < {upper_bound}"
            input_query = re.sub(date_range_pattern, date_query, input_query)
        return input_query

    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("No OPENAI_API_KEY provided in secrets.toml!")
        return None
    openai_api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    query = preprocess_date_range(query)

    if prompt_type == "pubmed":
        prompt = (
            f"Convert the following PubMed query into an **equivalent, explicit Scopus query**. "
            f"Ensure that the query strictly matches all conditions and avoids ambiguity. Use the following Scopus field codes:\n"
            f"- `AUTH` for author names (e.g., AUTH(\"LastName, FirstName\"))\n"
            f"- `AFFIL` for affiliations (e.g., AFFIL(\"Department of Physics\"))\n"
            f"- `AF-ID` for affiliation IDs (e.g., numeric IDs like 60004659)\n"
            f"- `TITLE` for document titles (e.g., TITLE(\"Quantum Computing\"))\n"
            f"- `DOI` for Digital Object Identifiers (e.g., DOI(\"10.1234/example.doi\"))\n"
            f"- `KEYWORDS` for keywords (e.g., KEYWORDS(\"Artificial Intelligence\"))\n"
            f"- `PUBYEAR` for publication year (e.g., PUBYEAR > 2015)\n"
            f"- `SOURCE` for journal source (e.g., SOURCE(\"Nature\"))\n\n"
            f"Special Instructions:\n"
            f"- **Always** connect different fields using `AND` to enforce strict matching.\n"
            f"- **Do not include unrelated records** that only partially match the criteria.\n"
            f"- If the query contains **only numeric input**, interpret it as an `AF-ID`.\n"
            f"- **Wrap the entire query in parentheses.**\n"
            f"- If an author's name is present, prioritize `AUTH` and `AF-ID` for accuracy.\n"
            f"- Use parentheses to group conditions logically, e.g., (AUTH(\"Smith\") AND AF-ID(12345)).\n\n"
            f"Query:\n\n{query}\n\n"
            f"Output the result as a **strict and explicitly formatted Scopus query only**."
        )
    elif prompt_type == "generic":
        prompt = (
            f"Convert the following unformatted query into a **strict and valid Scopus query**. "
            f"Ensure that the query explicitly matches all provided conditions. Use the following Scopus field codes:\n"
            f"- `AUTH` for author names (e.g., AUTH(\"LastName, FirstName\"))\n"
            f"- `AFFIL` for affiliations (e.g., AFFIL(\"Department of Physics\"))\n"
            f"- `AF-ID` for affiliation IDs (e.g., numeric IDs like 60004659)\n"
            f"- `TITLE` for document titles (e.g., TITLE(\"Quantum Computing\"))\n"
            f"- `DOI` for Digital Object Identifiers (e.g., DOI(\"10.1234/example.doi\"))\n"
            f"- `KEYWORDS` for keywords (e.g., KEYWORDS(\"Artificial Intelligence\"))\n"
            f"- `PUBYEAR` for publication year (e.g., PUBYEAR > 2015)\n"
            f"- `SOURCE` for journal source (e.g., SOURCE(\"Nature\"))\n\n"
            f"Special Instructions:\n"
            f"- **Always** connect different fields using `AND` to enforce strict matching.\n"
            f"- **Avoid** including unrelated records that only partially match the criteria.\n"
            f"- If the query contains **only numeric input**, interpret it as an `AF-ID`.\n"
            f"- **Wrap the entire query in parentheses.**\n"
            f"- Use parentheses to group conditions logically, e.g., (AUTH(\"Smith\") AND AF-ID(12345)).\n"
            f"- If multiple search fields are present, infer the fields intelligently based on their structure.\n\n"
            f"Query:\n\n{query}\n\n"
            f"Output the result as a **strict and explicitly formatted Scopus query only**."
        )
    elif prompt_type == "crossref":
        prompt = (
            f"Convert the following DOI into a valid CrossRef query format. "
            f"Ensure that the query is correctly formatted for CrossRef API calls. "
            f"Query:\n\n{query}\n\n"
            f"Output the result as a **strict and explicitly formatted CrossRef query only**."
        )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 150,
    }
    
    response = requests.post(f"{openai_api_base}/chat/completions", headers=headers, json=payload)
    
    if response.status_code != 200:
        st.error(f"LLM API call failed: {response.text}")
        return None
    
    result = response.json()
    if result and "choices" in result:
        raw_output = result["choices"][0]["message"]["content"].strip()
        cleaned_output = raw_output.strip("```")
        return cleaned_output
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
        #st.error(f"Error retrieving SNIP for ISSN {journal_issn}: {e}")
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
        st.error(f"Error executing Scopus query: {query}. {str(e)}")
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
    
@st.cache_data
def extract_dois_from_docx(file):
    """
    Extracts DOI numbers from a .docx file, removing "doi: " prefix and trailing periods.
    """
    try:
        document = Document(BytesIO(file.read()))
        text = " ".join([para.text.strip() for para in document.paragraphs])
        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
        raw_dois = re.findall(doi_pattern, text)
        cleaned_dois = [doi.rstrip(".") for doi in raw_dois]

        return list(set(cleaned_dois))  # Return unique DOIs
    except Exception as e:
        print(f"Error reading .docx file: {e}")
        return []

def is_crossref_available():
    """
    Check if the CrossRef API is functioning.
    """
    try:
        test_url = "https://api.crossref.org/works/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        response = requests.get(test_url, headers=headers)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data
def fetch_crossref_data(doi):
    """
    Query CrossRef for publication data using DOI.
    """
    if not is_crossref_available():
        st.warning("CrossRef API is not responding. Some data may be missing.")
        return None
    
    try:
        clean_doi = doi.strip().rstrip('.,;!?')
        encoded_doi = quote(clean_doi)
        url = f"https://api.crossref.org/works/{encoded_doi}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if "message" in data:
                return data["message"]
            else:
                st.warning("CrossRef response does not contain 'message' key.")
        elif response.status_code == 404:
            pass
        else:
            st.error(f"CrossRef API error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Error querying CrossRef for DOI {clean_doi}: {e}")
    return None

@st.cache_data
def fetch_data_for_dois(dois):
    total_dois = len(dois)
    progress_bar = st.progress(0)
    publication_data = []

    for index, doi in enumerate(dois, start=1):
        clean_doi = doi.strip()
        progress_bar.progress(index / total_dois)
        row_data = {}
        converted_query = convert_query_with_llm(f'DOI("{clean_doi}")', prompt_type="generic")
        if converted_query:
            scopus_data = fetch_scopus_data(converted_query)
            if not scopus_data.empty:
                row_data = scopus_data.iloc[0].to_dict()
        
        crossref_query = convert_query_with_llm(f'DOI("{clean_doi}")', prompt_type="crossref")
        if crossref_query:
            crossref_data = fetch_crossref_data(clean_doi)
            if crossref_data:
                row_data.update({
                    "journal_issn": row_data.get("journal_issn") or (crossref_data.get("ISSN", [None])[0] if isinstance(crossref_data.get("ISSN"), list) and crossref_data.get("ISSN") else None),
                    "publication_date": row_data.get("publication_date") or (crossref_data.get("issued", {}).get("date-parts", [[None]])[0][0] if crossref_data.get("issued") and crossref_data.get("issued").get("date-parts") else None),
                    "journal_name": row_data.get("journal_name") or (crossref_data.get("container-title", [None])[0] if isinstance(crossref_data.get("container-title"), list) and crossref_data.get("container-title") else None),
                    "title": row_data.get("title") or (crossref_data.get("title", [None])[0] if isinstance(crossref_data.get("title"), list) and crossref_data.get("title") else None),
                    "doi": row_data.get("doi") or crossref_data.get("DOI", None),
                    "author_names": row_data.get("author_names") or ", ".join([author.get("given", "") + " " + author.get("family", "") for author in crossref_data.get("author", [])]) if "author" in crossref_data else None,
                    "citation_count": row_data.get("citation_count") or crossref_data.get("is-referenced-by-count", None),
                    "publication_date": row_data.get("date_published") or crossref_data.get("created", {}).get("date-time", None),
                    "cited_by": crossref_data.get("is-referenced-by-count", None)  # Ensure the cited_by column is populated
                })

        if row_data:
            publication_data.append(row_data)
        else:
            st.warning(f"No results found for DOI: {clean_doi}")

    progress_bar.empty()

    if publication_data:
        return pd.DataFrame(publication_data)
    else:
        st.warning("No publication data found for the provided DOIs.")
        return pd.DataFrame(columns=["journal_issn", "publication_date", "journal_name", "title", "doi", "author_names", "citation_count", "date_published"])

# =============================================================================
# Data Processing Functions
# =============================================================================
# Build the co-author network
def normalize_name(name):
    """
    Normalize names to 'Firstname Lastname' format.
    """
    name = name.strip()
    if ',' in name:  # Format: "Lastname, Firstname"
        last, first = map(str.strip, name.split(',', maxsplit=1))
        return f"{first} {last}"
    return name  # Assume format is already "Firstname Lastname"

def build_coauthor_network(df):
    G = nx.Graph()

    for authors in df['author_names']:
        if authors:  # Ensure authors is not empty or invalid
            author_list = []
            
            # Handle semicolon-separated names first
            if ';' in authors:
                raw_authors = authors.split(';')  # Split by semicolon
            else:
                raw_authors = authors.split(',')  # Split by comma if no semicolons
            
            # Normalize all names to consistent format
            for raw_name in raw_authors:
                normalized_name = normalize_name(raw_name)
                if normalized_name:  # Ensure the name is valid
                    author_list.append(normalized_name)
            
            # Create edges for all combinations of authors
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
You can either upload a spreadsheet (e.g., a publication report), a word document with doi list, **or** enter and execute a Scopus query.
The app aggregates publications over time, enriches them with SNIP 
(Source-Normalized Impact per Paper) values, and allows for building plots to analyze publication stats.
""")

# ------------------------------------------------------------------------------
# Sidebar: API & Data Input Settings
# ------------------------------------------------------------------------------
st.sidebar.header("API & Data Input Settings")
data_source = st.sidebar.radio("Select Data Source", ["Scopus Query", "Upload Spreadsheet"])
df = pd.DataFrame()

if data_source == "Upload Spreadsheet":
    uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV, Excel, or DOCX)", type=["csv", "xls", "xlsx", "docx"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        
        if filename.endswith(('.csv', '.xls', '.xlsx')):
            df = load_data(uploaded_file)
            df = process_data(df)
        
        elif filename.endswith('.docx'):
            dois = extract_dois_from_docx(uploaded_file)
            if dois:
                df = fetch_data_for_dois(dois)
                if not df.empty:
                    df = process_data(df)
            else:
                st.error("No DOIs found in the uploaded DOCX file!")

elif data_source == "Scopus Query":
    st.sidebar.subheader("Enter Search Parameters")
    
    with st.sidebar.expander("LLM Query Conversion", expanded=False):
        conversion_method = st.selectbox(
            "Select Conversion Method",
            ["PubMed to Scopus Query", "Unformatted to Scopus Query"]
        )
        input_query = st.text_area("Enter your query", height=100, key="input_query")
        if st.button("Convert Query", key="llm_query_convert"):
            prompt_type = "pubmed" if conversion_method == "PubMed to Scopus Query" else "generic"
            converted_query = convert_query_with_llm(input_query, prompt_type=prompt_type)

            if converted_query:
                st.session_state.scopus_query = converted_query
            else:
                st.error("Conversion failed. Please check your input and API settings.")

    scopus_query = st.sidebar.text_area(
        "Enter Scopus Query Directly",
        value=st.session_state.get("scopus_query", ""),
        height=100,
        key="direct_query"
    )
    
    if st.sidebar.button("Execute Query", key="execute_scopus"):
        with st.spinner("Executing Scopus query..."):
            df = fetch_scopus_data(scopus_query)  # Fetch data
            if not df.empty:
                df = process_data(df)  # Process the data
                st.session_state.scopus_df = df

# ------------------------------------------------------------------------------
# Display Scopus Data
# ------------------------------------------------------------------------------
if "scopus_df" in st.session_state:
    df = st.session_state.scopus_df

# ------------------------------------------------------------------------------
# Main App: Display, Enrichment, and Visualization
# ------------------------------------------------------------------------------
current_year = datetime.datetime.now().year

if not df.empty:
    with st.spinner("Retrieving SNIP values from Elsevier..."):
        df = enrich_with_snip(df)

    desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
    other_columns = [col for col in df.columns if col not in desired_column_order]
    final_column_order = desired_column_order + other_columns
    df = df[final_column_order]
    df["MonthYear"] = df["Month"].astype(str) + "-" + df["Year"].astype(str)
    df["MonthYear"] = df["MonthYear"].str.replace(r"\.0", "", regex=True)
    df["MonthYear"] = pd.to_datetime(df["MonthYear"], format='%m-%Y', errors='coerce')
    df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    df['Year'] = df['Year'].astype('category')
    
    current_year = pd.Timestamp.now().year

    df_last_5_years = df[df['Year'].astype(float) >= (current_year - 4)]  # Use float for comparison

    st.subheader("Publications with Impact Factor (SNIP)")
    df = df.sort_values(by="SNIP", ascending=False)
    st.write(df)

    st.write("Click below to view all data via PyGWalker. ")
    if st.button("Click Here for Drag and Drop Interactive Data Viewer"):
        pyg.walk(df[["title", "Year", "MonthYear", "SNIP", "citedby_count"]], theme="dark")

    # Line Graph (last 5 years)
    monthly_counts_last_5_years, _ = aggregate_counts(df_last_5_years)
    grouped_counts_last_5_years = monthly_counts_last_5_years
    grouped_counts_last_5_years['Year'] = grouped_counts_last_5_years['Year'].astype(int)
    grouped_counts_last_5_years['Month'] = grouped_counts_last_5_years['Month'].astype(int)

    min_date = datetime.datetime(grouped_counts_last_5_years['Year'].min(), grouped_counts_last_5_years['Month'].min(), 1)
    max_date = datetime.datetime(grouped_counts_last_5_years['Year'].max(), grouped_counts_last_5_years['Month'].max(), 1)

    complete_date_range_last_5_years = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    date_range_df_last_5_years = pd.DataFrame({
        'YearMonth': complete_date_range_last_5_years.strftime('%Y-%m'),
        'Year': complete_date_range_last_5_years.year,
        'Month': complete_date_range_last_5_years.month
    })
    
    grouped_counts_last_5_years = pd.merge(date_range_df_last_5_years, grouped_counts_last_5_years, how='left', on=['Year', 'Month'])
    grouped_counts_last_5_years['Count'] = grouped_counts_last_5_years['Count'].fillna(0)
    grouped_counts_last_5_years['YearMonth'] = grouped_counts_last_5_years['YearMonth'].astype(str)
    
    grouped_counts_last_5_years = grouped_counts_last_5_years[
        grouped_counts_last_5_years['Year'] >= (current_year - 4)
    ]
    
    # Line Graph for the Last 5 Years
    st.subheader("Publication Trends (Last 5 Years)")
    fig = px.line(
        grouped_counts_last_5_years,  # Filtered data
        x='YearMonth',
        y='Count',
        markers=True,
        title="Monthly Publication Trend (Last 5 Years)"
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_traces(line=dict(color="darkgreen"), marker=dict(color="darkgreen"))
    
    fig.update_xaxes(
        tickmode='array',
        tickvals=grouped_counts_last_5_years['YearMonth'].iloc[::12],  # Label only filtered years
        tickangle=45
    )
    fig.update_layout(
        title=dict(
            text="Monthly Publication Trend (Last 5 Years)",
            font=dict(color="black", size=18),
            x=0.3
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
    )
    st.plotly_chart(fig, use_container_width=True, key="line_graph_last_5_years")
    
    # Violin Plot (last 5 years)
    fig = px.violin(
        df_last_5_years,
        x="Year",
        y="SNIP",
        box=True,
        points=False,
        title="SNIP Distribution by Year (Last 5 Years)"
    )
    fig.update_traces(marker_color="darkgreen")
    fig.update_layout(
        title=dict(
            text="SNIP Distribution by Year (Last 5 Years)",
            font=dict(color="black", size=18),
            x=0.3
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        height=600,
        width=800
    )
    st.plotly_chart(fig, use_container_width=True, key="violin_plot_last_5_years")

    def filter_network(graph, min_collaborations=1):
        filtered_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if data['weight'] >= min_collaborations:
                filtered_graph.add_node(u)
                filtered_graph.add_node(v)
                filtered_graph.add_edge(u, v, weight=data['weight'])
        return filtered_graph
    
    # Coauthor Network Visualisation
    if "author_names" in df_last_5_years.columns:
        st.write("### Co-Author Network Visualization (Last 5 Years)")
        
        # Build the co-author network
        coauthor_network = build_coauthor_network(df_last_5_years)
        
        # Filter by minimum collaborations dynamically using slider
        min_collaborations = st.slider("Minimum Collaborations to Display", 1, 10, 4)
        filtered_coauthor_network = filter_network(coauthor_network, min_collaborations=min_collaborations)
        
        # Visualize the graph
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(filtered_coauthor_network, seed=42, k=1.2, iterations=100)  # Adjusted k and iterations
        
        # Draw nodes with standard size
        nx.draw_networkx_nodes(
            filtered_coauthor_network, pos, node_size=200, node_color="green", alpha=0.8
        )
        
        # Draw edges with standard width
        nx.draw_networkx_edges(
            filtered_coauthor_network, pos, width=1.5, alpha=0.7, edge_color="gray"
        )
        
        # Add node labels
        nx.draw_networkx_labels(filtered_coauthor_network, pos, font_size=10, font_color="black")
        
        plt.title("Co-Author Network (Last 5 Years)", fontsize=14)
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