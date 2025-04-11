import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Import pybliometrics classes for API access.
from pybliometrics.scopus import SerialTitle, ScopusSearch, init, create_config

# =============================================================================
# pybliometrics Configuration
# =============================================================================
# Define the path where the pybliometrics config file should be created/located.
config_path = Path('./.config/pybliometrics.cfg')

# If the API key is provided via secrets.toml, set it as an environment variable
# and create the config file on the fly if needed.
if "SCOPUS_API_KEY" in st.secrets:
    api_key = st.secrets["SCOPUS_API_KEY"]
    os.environ["SCOPUS_API_KEY"] = api_key
    # Create (or update) the configuration file in the folder where the config will reside.
    # even though it is named "dir", the method wants the path. 
    create_config(config_dir=config_path, keys=[api_key])
else:
    st.warning("No SCOPUS_API_KEY provided in secrets.toml. Check your configuration.")

# Initialize the pybliometrics configuration (using our config file)
init(config_path=config_path)

# =============================================================================
# Global caching for SNIP lookups to help avoid duplicate API calls.
snip_cache = {}

def get_snip(journal_issn, pub_year):
    """
    Retrieve the SNIP (Source-Normalized Impact per Paper) for a given journal
    (by ISSN) and publication year using the Elsevier API via pybliometrics.
    Uses caching to avoid duplicate queries.

    Parameters:
        journal_issn (str): The ISSN (or E-ISSN) of the journal.
        pub_year (int): The publication year to retrieve the SNIP value for.

    Returns:
        float: The SNIP value if found; otherwise, np.nan.
    """
    key = (journal_issn, pub_year)
    if key in snip_cache:
        return snip_cache[key]
    if pd.isna(journal_issn) or str(journal_issn).strip() == "" or pd.isna(pub_year):
        snip_cache[key] = np.nan
        return np.nan
    try:
        # Retrieve the SerialTitle object with 'ENHANCED' view (which includes SNIP)
        st_obj = SerialTitle(str(journal_issn), refresh=True, view='ENHANCED')
        if st_obj.sniplist and len(st_obj.sniplist) > 0:
            # Look for an exact match in the SNIP list
            for yr, snip in st_obj.sniplist:
                if yr == pub_year:
                    snip_cache[key] = snip
                    return snip
            # If no exact match, choose the SNIP from the most recent year available
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
    """
    Load a publication file (CSV or Excel) into a Pandas DataFrame.
    The file is expected to include at least a 'publication_date' column.
    """
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
    """
    Process the raw publication data:
      - Convert the 'publication_date' column to datetime.
      - Create 'Year' and 'Month' columns.
    """
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df['Year'] = df['publication_date'].dt.year
    df['Month'] = df['publication_date'].dt.month
    return df

def fetch_scopus_data(query):
    """
    Execute a Scopus query using pybliometrics’ ScopusSearch and return the
    results as a DataFrame. Maps common fields to standard column names:
      - 'coverDate' → 'publication_date'
      - 'publicationName' → 'journal_name'
      - 'issn' → 'journal_issn'
    """
    try:
        search = ScopusSearch(query)
        if not search.results:
            st.error("No results found for the query!")
            return pd.DataFrame()
        df = pd.DataFrame(search.results)
        # Map and convert fields if they exist
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
    """
    Compute aggregated publication counts per Month (by Year and Month) and per Year.
    """
    monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    yearly_counts = df.groupby('Year').size().reset_index(name='Count')
    return monthly_counts, yearly_counts

@st.cache_data
def enrich_with_snip(df):
    """
    For each publication, retrieve the SNIP by performing a unique lookup per 
    (journal_issn, Year) pair. Adds a new column "SNIP" into the DataFrame.
    """
    unique_pairs = df[['journal_issn', 'Year']].drop_duplicates()
    snip_mapping = {}
    for _, row in unique_pairs.iterrows():
        issn = row['journal_issn']
        year = row['Year']
        snip_mapping[(issn, year)] = get_snip(issn, year)
    # Merge the SNIP values back to the original DataFrame.
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

# Select data source: file upload or Scopus query.
data_source = st.sidebar.radio("Select Data Source", ["Upload Spreadsheet", "Scopus Query"])

# DataFrame placeholder.
df = pd.DataFrame()

if data_source == "Upload Spreadsheet":
    uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = process_data(df)
elif data_source == "Scopus Query":
    scopus_query = st.sidebar.text_area("Enter Scopus Query", height=100)
    execute_query = st.sidebar.button("Execute Query")
    if execute_query:
        with st.spinner("Executing Scopus query..."):
            df = fetch_scopus_data(scopus_query)
            if not df.empty:
                df = process_data(df)
            st.session_state.scopus_df = df
    if "scopus_df" in st.session_state:
        df = st.session_state.scopus_df

# Main App: Display, Enrichment, and Visualization
if not df.empty:
    st.subheader("Raw Publication Data")
    st.write(df.head())

    with st.spinner("Retrieving SNIP values from Elsevier..."):
        df = enrich_with_snip(df)

    st.subheader("Enriched Data with SNIP")
    st.write(df.head())

    monthly_counts, yearly_counts = aggregate_counts(df)
    st.subheader("Publication Counts per Month")
    st.write(monthly_counts)
    st.subheader("Publication Counts per Year")
    st.write(yearly_counts)

    # Plot: Monthly Publication Trend using Plotly.
    st.markdown("### Monthly Publication Trend")
    monthly_counts['YearMonth'] = monthly_counts.apply(
        lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}", axis=1)
    fig1 = px.line(monthly_counts, x='YearMonth', y='Count', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Plot: SNIP Distribution (using Seaborn)
    st.markdown("### SNIP Distribution")
    group_by_option = st.radio("Group SNIP by:", ('Month', 'Year'))
    plot_type = st.sidebar.selectbox("Select Plot Type for SNIP Distribution", ("Violin Plot", "Box Plot"))

    plt.figure(figsize=(10, 6))
    if group_by_option == 'Month':
        if plot_type == "Violin Plot":
            sns.violinplot(x="Month", y="SNIP", data=df, inner="quartile")
        else:
            sns.boxplot(x="Month", y="SNIP", data=df)
        plt.title("SNIP Distribution by Month")
        plt.xlabel("Month")
    else:
        if plot_type == "Violin Plot":
            sns.violinplot(x="Year", y="SNIP", data=df, inner="quartile")
        else:
            sns.boxplot(x="Year", y="SNIP", data=df)
        plt.title("SNIP Distribution by Year")
        plt.xlabel("Year")
    plt.ylabel("SNIP")
    st.pyplot(plt.gcf())
    plt.clf()
    
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