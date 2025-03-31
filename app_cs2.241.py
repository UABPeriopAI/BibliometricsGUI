import streamlit as st
import pandas as pd
import numpy as np
import os
import pygwalker as pyg
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from itertools import combinations
from pathlib import Path
import datetime
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
        # Match date ranges like '2020-2025'
        date_range_pattern = r"(\d{4})-(\d{4})"
        match = re.search(date_range_pattern, input_query)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            # Adjust lower and upper bounds for strict matching
            lower_bound = start_year - 1
            upper_bound = end_year + 1
            # Replace the range with the PUBYEAR format
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

    # Preprocess the query to handle date ranges
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
    else:
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
# Sidebar: API & Data Input Settings
# ------------------------------------------------------------------------------
st.sidebar.header("API & Data Input Settings")
data_source = st.sidebar.radio("Select Data Source", ["Scopus Query", "Upload Spreadsheet"])
df = pd.DataFrame()

if data_source == "Upload Spreadsheet":
    # File upload functionality
    uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = process_data(df)

elif data_source == "Scopus Query":
    # Scopus Query functionality
    st.sidebar.subheader("Enter Search Parameters")
    
    # LLM Query Conversion expander
    with st.sidebar.expander("LLM Query Conversion", expanded=False):
        conversion_method = st.selectbox(
            "Select Conversion Method",
            ["PubMed to Scopus Query", "Unformatted to Scopus Query"]
        )
        input_query = st.text_area("Enter your query", height=100, key="input_query")  # Input field for LLM conversion
        if st.button("Convert Query", key="llm_query_convert"):
            prompt_type = "pubmed" if conversion_method == "PubMed to Scopus Query" else "generic"
            
            # Call function to convert the query
            converted_query = convert_query_with_llm(input_query, prompt_type=prompt_type)
            
            # Populate the "Enter Scopus Query Directly" text area with the converted query
            if converted_query:
                st.session_state.scopus_query = converted_query
            else:
                st.error("Conversion failed. Please check your input and API settings.")

    # Direct Scopus Query input field
    scopus_query = st.sidebar.text_area(
        "Enter Scopus Query Directly",
        value=st.session_state.get("scopus_query", ""),  # Populate with converted query if available
        height=100,
        key="direct_query"
    )
    
    # Button to execute the query
    if st.sidebar.button("Execute Query", key="execute_scopus"):
        # Use the query from the text area (either entered manually or populated by LLM conversion)
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
# Get the current year
current_year = datetime.datetime.now().year
if not df.empty:
    with st.spinner("Retrieving SNIP values from Elsevier..."):
        df = enrich_with_snip(df)

    # Reorder columns so key ones appear first
    desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
    other_columns = [col for col in df.columns if col not in desired_column_order]
    final_column_order = desired_column_order + other_columns
    df = df[final_column_order]
    
    # Combine Year and Month into a new column
    df["MonthYear"] = df["Month"].astype(str) + "-" + df["Year"].astype(str)
    
    # Convert 'MonthYear' to datetime format
    df["MonthYear"] = pd.to_datetime(df["MonthYear"], format='%m-%Y')  # Adjust format if needed (e.g., '%m-%Y' for month-year)
    
    # Ensure 'Year' values are valid (not exceeding available data)
    df['Year'] = df['Year'].astype('category')
    
    # Sort by SNIP values and display
    st.subheader("Publications with Impact Factor (SNIP)")
    df = df.sort_values(by="SNIP", ascending=False)
    st.write(df)
    
    # Launch PyGWalker (if button is clicked)
    if st.button("Click Here for Drag and Drop Interactive Data Viewer"):
        # Pass only the relevant fields to PyGWalker
        pyg.walk(df[["title", "Year", "MonthYear", "Month", "SNIP", "citedby_count"]], theme="dark")


    # Monthly Aggregate Counts
    monthly_counts, _ = aggregate_counts(df)
    grouped_counts = monthly_counts
    
    # Fill in missing months with zero counts
    grouped_counts['Year'] = grouped_counts['Year'].astype(int)
    grouped_counts['Month'] = grouped_counts['Month'].astype(int)
    
    # Create a complete date range
    from datetime import datetime
    min_date = datetime(grouped_counts['Year'].min(), grouped_counts['Month'].min(), 1)
    max_date = datetime(grouped_counts['Year'].max(), grouped_counts['Month'].max(), 1)
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    # Convert the date range to Year-Month format
    date_range_df = pd.DataFrame({
        'YearMonth': complete_date_range.strftime('%Y-%m'),
        'Year': complete_date_range.year,
        'Month': complete_date_range.month
    })
    
    # Merge with the grouped_counts to include missing months and fill with zeros
    grouped_counts = pd.merge(date_range_df, grouped_counts, how='left', on=['Year', 'Month'])
    grouped_counts['Count'] = grouped_counts['Count'].fillna(0)
    grouped_counts['YearMonth'] = grouped_counts['YearMonth'].astype(str)
    
    # Line Graph
    st.subheader("Publication Trends")
    fig = px.line(
        grouped_counts,
        x='YearMonth',
        y='Count',
        markers=True,
        title="Monthly Publication Trend"
    )
    fig.update_yaxes(rangemode="tozero")  # Start y-axis at zero
    fig.update_traces(line=dict(color="darkgreen"), marker=dict(color="darkgreen"))
    
    # Customize the x-axis date format and tick labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=grouped_counts['YearMonth'][::12],
        tickangle=45,
        tickformat='%b %Y'  # Format as "Month Year" (e.g., "Jan 2020")
    )
    
    # Add title and ensure a white background with black text
    fig.update_layout(
        title=dict(
            text="Monthly Publication Trend",  # Title text
            font=dict(color="black", size=18),  # Set title color and font size
            x=0.36  # Center the title
        ),
        plot_bgcolor="white",  # White background for the plot
        paper_bgcolor="white",  # White background for the figure
        font=dict(color="black", size=14),  # Black text for other elements
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
    )
    st.plotly_chart(fig, use_container_width=True, key="line_graph")


    
    # Violin Plot
    fig = px.violin(
        df,
        x="Year",
        y="SNIP",
        box=True,
        points=False,
        title="SNIP Distribution by Year",
    )
    fig.update_traces(marker_color="darkgreen") 
    
    # Add title and ensure a white background with black text
    fig.update_layout(
        title=dict(
            text="SNIP Distribution by Year",  # Title text
            font=dict(color="black", size=18),  # Set title color and font size
            x=0.36  # Center the title
        ),
        plot_bgcolor="white",  # White background for the plot
        paper_bgcolor="white",  # White background for the figure
        font=dict(color="black", size=14),  # Black text for other elements
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        height=600,  # Set the height of the figure
        width=800  # Set the width of the figure
    )
    st.plotly_chart(fig, use_container_width=True, key="violin_plot")

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
        edge_widths = [filtered_coauthor_network[u][v]['weight']/2 for u, v in filtered_coauthor_network.edges()]
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