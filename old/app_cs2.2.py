import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
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

if not df.empty:
    with st.spinner("Retrieving SNIP values from Elsevier..."):
        df = enrich_with_snip(df)

    # Ensure the specified column order appears first
    desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
    other_columns = [col for col in df.columns if col not in desired_column_order]
    final_column_order = desired_column_order + other_columns

    # Reorder the DataFrame columns
    df = df[final_column_order]

    # Display the enriched data with the specified column order and show the top 20 rows
    st.subheader("Publications with Impact Factor (SNIP)")
    df = df.sort_values(by="SNIP", ascending=False)
    st.write(df)

    # Create two columns for the sections
    col1, col2, col3 = st.columns(3)
    
    # Place "Aggregation Period" in the first column
    with col1:
        aggregation_option = st.radio("Select Aggregation Period", ["Monthly", "Quarterly", "Yearly"], key="aggregation_radio")
    
    # Place "Graph Type" in the second column
    with col2:
        graph_type = st.radio("Select Graph Type", ["Line", "Bar", "Scatter"], key="graph_type_radio")
        
    # Place "Theme" in the third column
    with col3:
        theme_option = st.radio("Select Theme", ["Dark", "Light"], key="theme_radio")
        
    # Customize appearance based on theme
    if theme_option == "Dark":
        plot_bgcolor = "black"
        paper_bgcolor = "black"
        font_color = "white"
        marker_color = "cyan"
    else:
        theme_option == "Light"
        plot_bgcolor = "white"
        paper_bgcolor = "white"
        font_color = "black"
        marker_color = "blue"

    if aggregation_option == "Monthly":
        monthly_counts, yearly_counts = aggregate_counts(df)
    
        # Create a complete DataFrame with all combinations of Year and Months
        all_months = pd.DataFrame({'Month': range(1, 13)})
        all_years = monthly_counts['Year'].unique()
        complete_monthly_counts = pd.MultiIndex.from_product([all_years, all_months['Month']], names=['Year', 'Month'])
        complete_df = pd.DataFrame(index=complete_monthly_counts).reset_index()
    
        # Ensure the Month column is numeric
        complete_df["Month"] = pd.to_numeric(complete_df["Month"], errors="coerce")
    
        # Merge to include the actual counts
        complete_df = complete_df.merge(monthly_counts, on=['Year', 'Month'], how='left').fillna(0)
        
        # Define corresponding abbreviations for each month
        month_abbreviations = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
    
        # Add a new column for plotting
        complete_df['YearMonth'] = complete_df['Year'].astype(str) + '-' + complete_df['Month'].apply(
            lambda x: month_abbreviations.get(int(x), "Unknown") if not pd.isna(x) else "Unknown"
        )
    
        # Filter the DataFrame to include only rows with nonzero values
        filtered_df = complete_df[complete_df['Count'] > 0]
    
        x_axis = "YearMonth"
        title = "Monthly Publication Trend"
        full_data = complete_df


    elif aggregation_option == "Quarterly":
        # Create a Quarter column from Month
        df['Quarter'] = df['Month'].apply(lambda x: (x - 1) // 3 + 1)

        # Group data by Year and Quarter, and count the occurrences
        quarterly_counts = df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')

        # Create a complete DataFrame with all combinations of Year and Quarters
        all_quarters = pd.DataFrame({'Quarter': range(1, 5)})
        all_years = quarterly_counts['Year'].unique()
        complete_quarterly_counts = pd.MultiIndex.from_product([all_years, all_quarters['Quarter']],
                                                               names=['Year', 'Quarter'])
        complete_df = pd.DataFrame(index=complete_quarterly_counts).reset_index()

        # Merge to include the actual counts
        complete_df = complete_df.merge(quarterly_counts, on=['Year', 'Quarter'], how='left').fillna(0)

        # Add a new column for plotting
        complete_df['YearQuarter'] = complete_df['Year'].astype(str) + '-' + complete_df['Quarter'].apply(
            lambda x: quarter_abbreviations[x])

        x_axis = "YearQuarter"
        title = "Quarterly Publication Trend"
        full_data = complete_df

    else:  # Yearly
        yearly_counts = df.groupby('Year').size().reset_index(name='Count')
        grouped_counts = yearly_counts
        grouped_counts['Year'] = grouped_counts['Year'].astype(str)

        x_axis = "Year"
        title = "Yearly Publication Trend"
        full_data = grouped_counts

    # Create the plot using Plotly
    if graph_type == "Bar":
        fig = px.bar(full_data, x=x_axis, y="Count", title=title, text="Count")
    elif graph_type == "Line":
        fig = px.line(full_data, x=x_axis, y="Count", title=title, markers=True)
    elif graph_type == "Scatter":
        fig = px.scatter(full_data, x=x_axis, y="Count", title=title)

    # Extract unique years for labeling
    unique_years = full_data[x_axis].apply(lambda x: x.split('-')[0]).unique()
    
    # Determine positions for the unique years on the x-axis
    tick_positions = [full_data[full_data[x_axis].str.startswith(year)].index[0] for year in unique_years]
    
    fig.update_layout(
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(color=font_color),  # General font color
        title_font=dict(size=20, color=font_color),  # Title font color
        title=dict(text=title, x=0.35),  # Center the title
        xaxis=dict(
            dtick=1,  # Ensure x-axis uses whole numbers for ticks
            tickvals=tick_positions,  # Positions of ticks
            ticktext=unique_years,  # Labels for the ticks
            tickfont=dict(color=font_color),  # X-axis text color
            title=dict(font=dict(color=font_color))  # X-axis title color
        ),
        yaxis=dict(
            dtick=1,  # Ensure y-axis uses whole numbers
            tickfont=dict(color=font_color),  # Y-axis text color
            title=dict(font=dict(color=font_color))  # Y-axis title color
        )
    )
    
    fig.update_traces(marker=dict(color=marker_color), textposition="top center")
    
    # Display the interactive plot
    st.plotly_chart(fig)


    st.markdown("### SNIP Distribution")
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Place the first radio button in the first column
    with col1:
        group_by_option = st.radio("Group SNIP by:", ('Month', 'Year'), key="group_by_radio")
    
    # Place the second radio button in the second column
    with col2:
        plot_type = st.radio("Select Plot Type for SNIP Distribution", ("Violin Plot", "Box Plot"), key="plot_type_radio")
    
    # Map numeric months to their abbreviations
    month_mapping = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                     7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    
    # Replace numeric month labels with abbreviations
    df["Month"] = df["Month"].map(month_mapping)
    
    # Combine Month and Year into a single column for unique representation
    df["Month-Year"] = df["Month"] + " " + df["Year"].astype(str)
    
    # Parse Month-Year as datetime for proper sorting
    df["Month-Year-Date"] = pd.to_datetime(df["Month-Year"], format="%b %Y")
    
    # Sort the DataFrame by Month-Year-Date to ensure chronological order
    df = df.sort_values("Month-Year-Date")

    # Create an interactive Plotly plot
    if group_by_option == 'Month':
        x_axis = "Month-Year"
        title = "SNIP Distribution by Month-Year"
    else:
        x_axis = "Year"
        title = "SNIP Distribution by Year"
    
    # Use Plotly Express to create the interactive plot
    if plot_type == "Violin Plot":
        fig = px.violin(df, x=x_axis, y="SNIP", box=True, points="all", title=title, template="plotly_dark")
    elif plot_type == "Box Plot":
        fig = px.box(df, x=x_axis, y="SNIP", title=title, template="plotly_dark")
    
    # Customize the appearance of the plot
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        title_font=dict(size=20, color="white"),
        xaxis=dict(title=x_axis, tickangle=45, title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(title="SNIP", title_font=dict(color="white"), tickfont=dict(color="white"))
    )
    
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

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