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
    st.write(df)  # Adjusted to display the top 20 rows

    # Select aggregation type
    aggregation_option = st.radio("Select Aggregation Period", ["Monthly", "Quarterly", "Yearly"])

    if aggregation_option == "Monthly":
        monthly_counts, yearly_counts = aggregate_counts(df)
        
        # Create a complete DataFrame with all combinations of Year and Months
        all_months = pd.DataFrame({'Month': range(1, 13)})
        all_years = monthly_counts['Year'].unique()
        complete_monthly_counts = pd.MultiIndex.from_product([all_years, all_months['Month']], names=['Year', 'Month'])
        complete_df = pd.DataFrame(index=complete_monthly_counts).reset_index()
        
        # Merge to include the actual counts
        complete_df = complete_df.merge(monthly_counts, on=['Year', 'Month'], how='left').fillna(0)
        
        # Create a pivot table
        pivot_table = complete_df.pivot_table(index='Year', columns='Month', values='Count', fill_value=0)
        
        # Define corresponding abbreviations for each month
        month_abbreviations = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        # Reindex the pivot table to ensure all months are included
        pivot_table = pivot_table.reindex(columns=range(1, 13), fill_value=0)
        
        # Rename columns to use month abbreviations
        pivot_table.columns = [month_abbreviations[col] for col in pivot_table.columns]
        
        # Add a total column
        pivot_table['Total'] = pivot_table.sum(axis=1)
        
        # Create a display table with zeros replaced by a period ('.')
        display_table = pivot_table.copy()  # Create a copy to preserve the original data
        display_table = display_table.replace(0, " ")  # Replace zeros with a dot
        
        # Display the modified table
        st.subheader("Publication Counts Monthly (No Zeros Shown)")
        st.write(display_table)
        
        # Add a new column for plotting
        complete_df['YearMonth'] = complete_df['Year'].astype(str) + '-' + complete_df['Month'].apply(lambda x: month_abbreviations[x])
        
        # Filter the DataFrame to include only rows with nonzero values
        filtered_df = complete_df[complete_df['Count'] > 0]

    
    elif aggregation_option == "Quarterly":
        # Create a Quarter column from Month
        df['Quarter'] = df['Month'].apply(lambda x: (x - 1) // 3 + 1)
    
        # Group data by Year and Quarter, and count the occurrences
        quarterly_counts = df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')
    
        # Create a complete DataFrame with all combinations of Year and Quarters
        all_quarters = pd.DataFrame({'Quarter': range(1, 5)})  # Quarters: Q1 to Q4
        all_years = quarterly_counts['Year'].unique()
        complete_quarterly_counts = pd.MultiIndex.from_product([all_years, all_quarters['Quarter']], names=['Year', 'Quarter'])
        complete_df = pd.DataFrame(index=complete_quarterly_counts).reset_index()
    
        # Merge to include the actual counts
        complete_df = complete_df.merge(quarterly_counts, on=['Year', 'Quarter'], how='left').fillna(0)
    
        # Create a pivot table
        pivot_table = complete_df.pivot_table(index='Year', columns='Quarter', values='Count', fill_value=0)
    
        # Rename columns to use Quarter names
        quarter_abbreviations = {
            1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'
        }
        pivot_table.columns = [quarter_abbreviations[col] for col in pivot_table.columns]
    
        # Add a Total column
        pivot_table['Total'] = pivot_table.sum(axis=1)
    
        # Display the pivot table
        st.subheader("Publication Counts Quarterly")
        st.write(pivot_table)

        # Add a new column for plotting
        complete_df['YearQuarter'] = complete_df['Year'].astype(str) + '-' + complete_df['Quarter'].apply(lambda x: quarter_abbreviations[x])

    else:  # Yearly
        yearly_counts = df.groupby('Year').size().reset_index(name='Count')
        grouped_counts = yearly_counts
        grouped_counts['Year'] = grouped_counts['Year'].astype(str)
    
        # Display the yearly counts
        st.subheader("Publication Counts Yearly")
        st.write(grouped_counts)

    # Select Graph Type
    graph_type = st.radio("Select Graph Type", ["Line", "Bar", "Scatter"])

    # Filter the data to match aggregation_option
    if aggregation_option == "Monthly":
        x_axis = "YearMonth"
        title = "Monthly Publication Trend"
        full_data = complete_df  # Use full dataset
    elif aggregation_option == "Quarterly":
        x_axis = "YearQuarter"
        title = "Quarterly Publication Trend"
        full_data = complete_df
    else:  # Yearly
        x_axis = "Year"
        title = "Yearly Publication Trend"
        full_data = grouped_counts
    
    # Initialize the figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot the full dataset
    if graph_type == "Bar":
        sns.barplot(x=x_axis, y="Count", data=full_data, palette="coolwarm")
    elif graph_type == "Line":
        sns.lineplot(x=x_axis, y="Count", data=full_data, marker="o", color="cyan")
    elif graph_type == "Scatter":
        plt.scatter(full_data[x_axis], full_data["Count"], color="cyan")
    
    # Set the x-axis labels to show every 3rd label
    plt.xticks(
        ticks=range(len(full_data[x_axis])),
        labels=[label if i % 3 == 0 else "" for i, label in enumerate(full_data[x_axis])],
        rotation=45,
        color="white"
    )
    
    # Customize plot appearance
    plt.title(title, fontsize=16, fontweight="bold", color="white")
    plt.xlabel(x_axis, fontsize=12, color="white")
    plt.ylabel("Count", fontsize=12, color="white")
    plt.yticks(color="white")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    
    # Set dark theme for the plot
    plt.gca().set_facecolor("black")
    fig.patch.set_facecolor("black")
    plt.gca().spines["top"].set_color("white")
    plt.gca().spines["right"].set_color("white")
    plt.gca().spines["left"].set_color("white")
    plt.gca().spines["bottom"].set_color("white")
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    # SNIP Distribution Visualization
    st.markdown("### SNIP Distribution")
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Place the first radio button in the first column
    with col1:
        group_by_option = st.radio("Group SNIP by:", ('Month', 'Year'))
    
    # Place the second radio button in the second column
    with col2:
        plot_type = st.radio("Select Plot Type for SNIP Distribution", ("Violin Plot", "Box Plot"))
        
    # Set the Seaborn theme to "dark"
    sns.set_theme(style="ticks")  # You can switch to "dark" for no gridlines
    
    # Adjust Matplotlib to have a dark background
    plt.rcParams['axes.facecolor'] = 'black'    # Background of the plot area
    plt.rcParams['axes.edgecolor'] = 'black'   # Edge color of the plot area
    plt.rcParams['figure.facecolor'] = 'black' # Background of the entire figure
    plt.rcParams['xtick.color'] = 'white'      # X-axis tick color
    plt.rcParams['ytick.color'] = 'white'      # Y-axis tick color
    plt.rcParams['text.color'] = 'white'       # General text color (titles, labels, etc.)
    plt.rcParams['axes.labelcolor'] = 'white'  # X and Y label color
    plt.rcParams['axes.titlecolor'] = 'white'  # Title color
    
    # Map numeric months to their abbreviations
    month_mapping = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                     7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    
    # Replace numeric month labels with abbreviations
    df["Month"] = df["Month"].map(month_mapping)
    
    st.markdown("### SNIP Distribution")
    plt.figure(figsize=(10, 6))
    
    # Combine Month and Year into a single column for unique representation
    df["Month-Year"] = df["Month"] + " " + df["Year"].astype(str)  # Create a new column
    
    if group_by_option == 'Month':
        if plot_type == "Violin Plot":
            # Enhanced violin plot for Month-Year
            sns.violinplot(x="Month-Year", y="SNIP", data=df, inner="quartile", palette="coolwarm")
            sns.stripplot(x="Month-Year", y="SNIP", data=df, color="white", size=3, jitter=True)  # Adds jitter for clarity
        else:
            sns.boxplot(x="Month-Year", y="SNIP", data=df, palette="coolwarm")
        plt.title("SNIP Distribution by Month-Year")
        plt.xlabel("Month-Year")
    
    else:
        if plot_type == "Violin Plot":
            sns.violinplot(x="Year", y="SNIP", data=df, inner="quartile", palette="coolwarm")
            sns.stripplot(x="Year", y="SNIP", data=df, color="white", size=3, jitter=True)
        else:
            sns.boxplot(x="Year", y="SNIP", data=df, palette="coolwarm")
        plt.title("SNIP Distribution by Year")
        plt.xlabel("Year")
    
    plt.xticks(rotation=45, fontsize=10, color="white")
    plt.yticks(color="white")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.gca().set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")
    plt.ylabel("SNIP", fontsize=12, color="white")
    
    # Display the plot
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