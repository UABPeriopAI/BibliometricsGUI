import streamlit as st
import pandas as pd
import numpy as np
import os
import pygwalker as pyg
from pathlib import Path
from pybliometrics.scopus import SerialTitle, ScopusSearch, init, create_config

# =============================================================================
# pybliometrics Configuration
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
# Global caching for SNIP lookups to help avoid duplicate API calls.
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
# Streamlit App Layout and Main Logic
# =============================================================================
st.title("Publication Metrics Dashboard")
st.markdown("""
This app allows you to explore publication metrics for the Division of Molecular and Translational BioMedicine.
You can either upload a spreadsheet (e.g., a publication report) **or** enter and execute a Scopus query.
The app aggregates publications over time and, by using pybliometrics’ SerialTitle API, retrieves SNIP
(Source-Normalized Impact per Paper) values. Duplicate SNIP lookups for the same journal and year are avoided.
""")

# Sidebar: API & Data Input Settings
st.sidebar.header("API & Data Input Settings")
data_source = st.sidebar.radio("Select Data Source", ["Upload Spreadsheet", "Scopus Query"])

# DataFrame placeholder.
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
    institution = st.sidebar.text_input("Affiliation Institution")  # Box for institution
    title = st.sidebar.text_input("Title")
    journal = st.sidebar.text_input("Journal")
    scopus_query = st.sidebar.text_area("Or Enter Scopus Query Directly", height=100)

    def format_scopus_query(author, start_year, end_year, title, journal, institution):
        query = ""
        if author:
            query += f"AUTH({author}) AND "
        if start_year and end_year:
            query += f"PUBYEAR AFT {start_year} AND PUBYEAR BEF {end_year} AND "
        if institution:
            query += f"AFFIL({institution}) AND "
        if title:
            query += f"TITLE({title}) AND "
        if journal:
            query += f"SRCTITLE({journal}) AND "
        query = query.rstrip(" AND ")
        return query

    if st.sidebar.button("Execute Query"):
        if not scopus_query:
            scopus_query = format_scopus_query(author, start_year, end_year, title, journal, institution)
        with st.spinner("Executing Scopus query..."):
            df = fetch_scopus_data(scopus_query)
            if not df.empty:
                df = process_data(df)
                st.session_state.scopus_df = df

if "scopus_df" in st.session_state:
    df = st.session_state.scopus_df

# Main App: Display, Enrichment, and Visualization
    # Ensure your DataFrame is populated
    if not df.empty:
        with st.spinner("Retrieving SNIP values from Elsevier..."):
            df = enrich_with_snip(df)
    
        # Ensure the specified column order
        desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
        other_columns = [col for col in df.columns if col not in desired_column_order]
        final_column_order = desired_column_order + other_columns
        df = df[final_column_order]
    
        # Display enriched data
        st.subheader("Publications with Impact Factor (SNIP)")
        df = df.sort_values(by="SNIP", ascending=False)
        st.write(df)
    
        # Use PyGWalker to create the visualizations
        st.markdown("### Interactive Data Exploration with PyGWalker")
        walker = pyg.walk(df)
    
        # Display PyGWalker visualization in Streamlit
        st.write(walker)
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
• Duplicate lookups for the same journal and year are avoided by performing a unique query per (journal_issn, Year) pair.  
• Ensure your Elsevier API key is provided in Streamlit’s secrets.toml so that the config file can be created automatically.
""")