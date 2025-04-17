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
from urllib.parse import quote
from pybliometrics.scopus import SerialTitle, ScopusSearch, init, create_config

# =============================================================================
# Configuration Manager Class
# =============================================================================
class ConfigManager:
    @staticmethod
    def setup_pybliometrics(config_path, scopus_api_key):
        if scopus_api_key:
            os.environ["SCOPUS_API_KEY"] = scopus_api_key
            create_config(config_dir=config_path, keys=[scopus_api_key])
        else:
            st.warning("No SCOPUS_API_KEY provided. Check your configuration.")
        init(config_path=config_path)

    @staticmethod
    def get_openai_headers(api_key):
        if not api_key:
            st.error("No OPENAI_API_KEY provided!")
            return None
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# =============================================================================
# Query Conversion Class
# =============================================================================
class QueryConverter:
    @staticmethod
    def preprocess_date_range(input_query):
        """
        Detect and process date ranges within the input query.
        Returns a reformatted string with consistent PUBYEAR conditions.
        """
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

    @staticmethod
    def convert_query(query, prompt_type, api_headers, openai_api_base):
        """
        Convert a query using a prompt and OpenAI API.
        Handles multiple query types (PubMed, generic, or CrossRef).

        Args:
            query (str): The input query to be converted.
            prompt_type (str): The type of query (e.g., "pubmed", "generic", "crossref").
            api_headers (dict): API headers for the OpenAI API.
            openai_api_base (str): Base URL for the OpenAI API.

        Returns:
            str: The converted query or None if conversion fails.
        """
        query = QueryConverter.preprocess_date_range(query)

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
                f"- `PUBYEAR` for publication year (e.g., PUBYEAR = 2024)\n"
                f"- `SOURCE` for journal source (e.g., SOURCE(\"Nature\"))\n\n"
                f"Special Instructions:\n"
                f"- **Always** connect different fields using `AND` to enforce strict matching.\n"
                f"- **Do not include unrelated records** that only partially match the criteria.\n"
                f"- If the query contains **only numeric input**, interpret it as an `AF-ID` (numeric values should not have quotes).\n"
                f"- **Wrap the entire query in parentheses.**\n"
                f"- If an author's name is present, prioritize `AUTH` and `AF-ID` for accuracy.\n"
                f"- Use parentheses to group conditions logically, e.g., (AUTH(\"Smith\") AND AF-ID(12345)).\n"
                f"- For single PUBYEAR ranges, always use `=` (e.g., PUBYEAR = 2024).\n\n"
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
                f"- `PUBYEAR` for publication year (e.g., PUBYEAR = 2024)\n"
                f"- `SOURCE` for journal source (e.g., SOURCE(\"Nature\"))\n\n"
                f"Special Instructions:\n"
                f"- **Always** connect different fields using `AND` to enforce strict matching.\n"
                f"- **Avoid** including unrelated records that only partially match the criteria.\n"
                f"- If the query contains **only numeric input**, interpret it as an `AF-ID` (numeric values should not have quotes).\n"
                f"- **Wrap the entire query in parentheses.**\n"
                f"- Use parentheses to group conditions logically, e.g., (AUTH(\"Smith\") AND AF-ID(12345)).\n"
                f"- For single PUBYEAR ranges, always use `=` (e.g., PUBYEAR = 2024).\n\n"
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
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        # API Request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150
        }
        try:
            response = requests.post(f"{openai_api_base}/chat/completions", headers=api_headers, json=payload)
            response.raise_for_status()  # Raise an error for HTTP codes like 4XX/5XX
            response_data = response.json()
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"].strip().strip("```")
        except requests.RequestException as e:
            st.error(f"Error during OpenAI API request: {e}")
        return None


# =============================================================================
# CrossRef Manager Class
# =============================================================================
class CrossRefManager:
    @staticmethod
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
        # Correctly referenced static method from CrossRefManager class
        if not CrossRefManager.is_crossref_available():
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
    
    @staticmethod
    @st.cache_data
    def fetch_data_for_dois(dois, api_headers, openai_api_base):
        """
        Query CrossRef and Scopus for publication data using DOIs.

        Args:
            dois (list): List of DOIs to query.
            api_headers (dict): API headers for the OpenAI API.
            openai_api_base (str): Base URL for the OpenAI API.

        Returns:
            pd.DataFrame: A DataFrame containing publication data.
        """
        total_dois = len(dois)
        progress_bar = st.progress(0)
        publication_data = []

        for index, doi in enumerate(dois, start=1):
            clean_doi = doi.strip()
            progress_bar.progress(index / total_dois)
            row_data = {}

            # Convert query using QueryConverter for Scopus data
            converted_query = QueryConverter.convert_query(
                f'DOI("{clean_doi}")',
                prompt_type="generic",
                api_headers=api_headers,
                openai_api_base=openai_api_base
            )
            if converted_query:
                scopus_data = DataProcessor.fetch_scopus_data(converted_query)
                if not scopus_data.empty:
                    row_data = scopus_data.iloc[0].to_dict()

            # Convert query using QueryConverter for CrossRef data
            crossref_query = QueryConverter.convert_query(
                f'DOI("{clean_doi}")',
                prompt_type="crossref",
                api_headers=api_headers,
                openai_api_base=openai_api_base
            )
            if crossref_query:
                crossref_data = CrossRefManager.fetch_crossref_data(clean_doi)
                if crossref_data:
                    row_data.update({
                        "journal_issn": row_data.get("journal_issn") or (
                            crossref_data.get("ISSN", [None])[0] if isinstance(crossref_data.get("ISSN"), list) and crossref_data.get("ISSN") else None
                        ),
                        "publication_date": row_data.get("publication_date") or (
                            crossref_data.get("issued", {}).get("date-parts", [[None]])[0][0] if crossref_data.get("issued") and crossref_data.get("issued").get("date-parts") else None
                        ),
                        "journal_name": row_data.get("journal_name") or (
                            crossref_data.get("container-title", [None])[0] if isinstance(crossref_data.get("container-title"), list) and crossref_data.get("container-title") else None
                        ),
                        "title": row_data.get("title") or (
                            crossref_data.get("title", [None])[0] if isinstance(crossref_data.get("title"), list) and crossref_data.get("title") else None
                        ),
                        "doi": row_data.get("doi") or crossref_data.get("DOI", None),
                        "author_names": row_data.get("author_names") or ", ".join([
                            author.get("given", "") + " " + author.get("family", "")
                            for author in crossref_data.get("author", [])
                        ]) if "author" in crossref_data else None,
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
    
        progress_bar.empty()
    
        if publication_data:
            return pd.DataFrame(publication_data)
        else:
            st.warning("No publication data found for the provided DOIs.")
            return pd.DataFrame(columns=["journal_issn", "publication_date", "journal_name", "title", "doi", "author_names", "citation_count", "date_published"])


# =============================================================================
# SNIP Manager Class
# =============================================================================
class SNIPManager:
    snip_cache = {}  # Class-level cache for SNIP values

    @staticmethod
    def get_snip(journal_issn, pub_year):
        key = (journal_issn, pub_year)
        if key in SNIPManager.snip_cache:  # Corrected reference to class-level snip_cache
            return SNIPManager.snip_cache[key]
        if pd.isna(journal_issn) or str(journal_issn).strip() == "" or pd.isna(pub_year):
            SNIPManager.snip_cache[key] = np.nan  # Corrected reference to class-level snip_cache
            return np.nan
        try:
            st_obj = SerialTitle(str(journal_issn), refresh=True, view='ENHANCED')
            if st_obj.sniplist and len(st_obj.sniplist) > 0:
                for yr, snip in st_obj.sniplist:
                    if yr == pub_year:
                        SNIPManager.snip_cache[key] = snip  # Corrected reference to class-level snip_cache
                        return snip
                latest_snip = max(st_obj.sniplist, key=lambda x: x[0])[1]
                SNIPManager.snip_cache[key] = latest_snip  # Corrected reference to class-level snip_cache
                return latest_snip
            else:
                SNIPManager.snip_cache[key] = np.nan  # Corrected reference to class-level snip_cache
                return np.nan
        except Exception as e:
            # st.error(f"Error retrieving SNIP for ISSN {journal_issn}: {e}")
            SNIPManager.snip_cache[key] = np.nan  # Corrected reference to class-level snip_cache
            return np.nan


# =============================================================================
# Data Processor Class
# =============================================================================
class DataProcessor:
    @staticmethod
    def fetch_scopus_data(query):
        """
        Execute a Scopus query and return the results as a DataFrame.

        Args:
            query (str): The Scopus query string.

        Returns:
            pd.DataFrame: DataFrame containing Scopus search results.
        """
        try:
            # Perform Scopus search using ScopusSearch from pybliometrics
            search = ScopusSearch(query)
            if not search.results:
                return pd.DataFrame()  # Return an empty DataFrame if no results are found
            
            # Convert results to a DataFrame
            df = pd.DataFrame(search.results)
            
            # Process and add relevant columns
            if 'coverDate' in df.columns:
                df['publication_date'] = pd.to_datetime(df['coverDate'], errors='coerce')
            if 'publicationName' in df.columns:
                df['journal_name'] = df['publicationName']
            if 'issn' in df.columns:
                df['journal_issn'] = df['issn']

            return df
        except Exception as e:
            # Provide feedback for debugging or errors
            st.error(f"Error executing Scopus query: {query}. {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame on error
        
    @staticmethod
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        elif file.name.endswith('.docx'):
            return DataProcessor.extract_dois_from_docx(file)
        return pd.DataFrame()

    @staticmethod
    @st.cache_data
    def process_data(df):
        # Ensure 'publication_date' is in datetime format
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    
        # Extract 'Year' and 'Month' columns
        df['Year'] = df['publication_date'].dt.year
        df['Month'] = df['publication_date'].dt.month
    
        # Combine 'Year' and 'Month' to create 'MonthYear'
        df['MonthYear'] = df['Month'].astype(str) + "-" + df['Year'].astype(str)
    
        return df

    @staticmethod
    def extract_dois_from_docx(file):
        document = Document(BytesIO(file.read()))
        text = " ".join(para.text.strip() for para in document.paragraphs)
        return list(set(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)))
        
    
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
    
    @staticmethod
    @st.cache_data
    def enrich_with_snip(df):
        """
        Enrich the DataFrame with SNIP values using journal ISSN and publication year.
        """
        unique_pairs = df[['journal_issn', 'Year']].drop_duplicates()
        snip_mapping = {}
    
        for _, row in unique_pairs.iterrows():
            issn = row['journal_issn']
            year = row['Year']
            # Use the fully qualified name to call `get_snip`
            snip_mapping[(issn, year)] = SNIPManager.get_snip(issn, year)
    
        # Apply SNIP values to the DataFrame
        df['SNIP'] = df.apply(lambda row: snip_mapping.get((row['journal_issn'], row['Year']), np.nan), axis=1)
        return df


# =============================================================================
# Network Builder Class
# =============================================================================
class NetworkBuilder:
    @staticmethod
    def normalize_name(name):
        """
        Normalize names to 'Firstname Lastname' format.
        """
        name = name.strip()
        if ',' in name:  # Format: "Lastname, Firstname"
            last, first = map(str.strip, name.split(',', maxsplit=1))
            return f"{first} {last}"
        return name  # Assume format is already "Firstname Lastname"
    
    @staticmethod
    def build_coauthor_network(df):
        """
        Build a coauthor network from a DataFrame containing author names.

        Args:
            df (pd.DataFrame): DataFrame with an 'author_names' column.

        Returns:
            networkx.Graph: A graph where nodes represent authors and edges represent collaborations.
        """
        # Validate input
        if 'author_names' not in df.columns:
            raise ValueError("The DataFrame must contain an 'author_names' column.")
        
        G = nx.Graph()

        for authors in df['author_names']:
            if authors and isinstance(authors, str):  # Ensure non-empty and valid string
                # Normalize names and split by ';'
                names = [NetworkBuilder.normalize_name(name) for name in authors.split(';') if name.strip()]
                # Generate edges for all pairs of authors
                for pair in combinations(names, 2):
                    if G.has_edge(*pair):
                        G[pair[0]][pair[1]]['weight'] += 1
                    else:
                        G.add_edge(*pair, weight=1)

        return G

# =============================================================================
# Main App Class
# =============================================================================
class MetricsAppBase:
    def handle_uploaded_file(self, file):
        if not hasattr(file, 'name') or not isinstance(file.name, str):
            st.error("Invalid file uploaded. Please try again.")
            return pd.DataFrame()
    
        filename = file.name.lower()
        try:
            if filename.endswith(('.csv', '.xls', '.xlsx')):
                df = DataProcessor.load_data(file)
                processed_df = DataProcessor.process_data(df)
                st.session_state["scopus_df"] = processed_df  # Persist processed DataFrame
                return processed_df
            elif filename.endswith('.docx'):
                dois = DataProcessor.extract_dois_from_docx(file)
                if dois:
                    raw_df = CrossRefManager.fetch_data_for_dois(dois, self.api_headers, self.openai_api_base)
                    processed_df = DataProcessor.process_data(raw_df)
                    st.session_state["scopus_df"] = processed_df  # Persist processed DataFrame
                    return processed_df
                else:
                    st.error("No DOIs found in the uploaded DOCX file!")
            else:
                st.error("Unsupported file type. Please upload a valid CSV, Excel, or DOCX file.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
        return pd.DataFrame()
        
class BasicMetricsApp(MetricsAppBase):
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_year = datetime.datetime.now().year

    def run(self):
        """
        Main entry point for the Streamlit app.
        """
        st.title("Publication Metrics Dashboard")
        st.markdown("""
        This app allows you to explore publication metrics for the Division of Molecular and Translational BioMedicine.
        You can either upload a spreadsheet (e.g., a publication report), a word document with DOI list, **or** enter and execute a Scopus query.
        The app aggregates publications over time, enriches them with SNIP 
        (Source-Normalized Impact per Paper) values, and allows for building plots to analyze publication stats.
        """)

        # Sidebar: API & Data Input Settings
        st.sidebar.header("API & Data Input Settings")
        data_source = st.sidebar.radio("Select Data Source", ["Scopus Query", "Upload Spreadsheet"])
        df = pd.DataFrame()

        if data_source == "Upload Spreadsheet":
            uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV, Excel, or DOCX)", type=["csv", "xls", "xlsx", "docx"])
            if uploaded_file:
                df = self.handle_uploaded_file(uploaded_file)

        elif data_source == "Scopus Query":
            st.sidebar.subheader("Enter Search Parameters")

            # Sidebar widget logic
            with st.sidebar.expander("LLM Query Conversion", expanded=False):
                conversion_method = st.selectbox(
                    "Select Conversion Method",
                    ["Unformatted to Scopus Query", "PubMed to Scopus Query"]
                )
                input_query = st.text_area("Enter your query", height=100, key="input_query")
            
                if st.button("Convert Query", key="llm_query_convert"):
                    prompt_type = "pubmed" if conversion_method == "PubMed to Scopus Query" else "generic"
                    converted_query = QueryConverter.convert_query(st.session_state.input_query, prompt_type, self.api_headers, self.openai_api_base)
            
                    if converted_query:
                        st.session_state.scopus_query = converted_query
                    else:
                        st.error("Conversion failed. Possible reasons: malformed input or API issues.")
            
            # Direct Scopus query input and execution
            scopus_query = st.sidebar.text_area(
                "Enter Scopus Query Directly",
                value=st.session_state.get("scopus_query", ""),
                height=100,
                key="direct_query"
            )
            
            if st.sidebar.button("Execute Query", key="execute_scopus"):
                with st.spinner("Executing Scopus query..."):
                    self.df = DataProcessor.fetch_scopus_data(st.session_state.scopus_query)
                    if self.df.empty:
                        st.warning("No data found for the Scopus query. Please refine your query.")
                    else:
                        self.df = DataProcessor.process_data(self.df)
                        st.session_state["scopus_df"] = self.df  # Persist DataFrame to session state
                        
# ------------------------------------------------------------------------------
# Display Scopus Data
# ------------------------------------------------------------------------------
class AdvancedMetricsApp(MetricsAppBase):
    def __init__(self):
        self.df = pd.DataFrame()
        self.current_year = datetime.datetime.now().year
        self.config_path = Path('./.config/pybliometrics.cfg')
        self.scopus_api_key = st.secrets.get("SCOPUS_API_KEY")
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY")
        self.openai_api_base = st.secrets.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    
        # Setup Pybliometrics configuration
        ConfigManager.setup_pybliometrics(self.config_path, self.scopus_api_key)
        self.api_headers = ConfigManager.get_openai_headers(self.openai_api_key)
    
        # Initialize session state variables
        st.session_state.setdefault("scopus_query", "")
        st.session_state.setdefault("scopus_df", pd.DataFrame())
        st.session_state.setdefault("pygwalker_html", "")
        st.session_state.setdefault("input_query", "")

    
    @st.cache_data
    def generate_pygwalker_html(df):
        return pyg.walk(df[["title", "Year", "MonthYear", "SNIP", "citedby_count"]], return_html=True)
    
    def display_scopus_data(self):
        st.write("Click below to toggle the pygwalker data viewer. This allows you to create visualizations based on the data provided above.")
        # Option to toggle PyGWalker Viewer
        if st.button("Show PyGWalker Viewer"):
            try:
                # Generate PyGWalker visualization
                walker_html = pyg.walk(
                    self.df[["title", "Year", "MonthYear", "SNIP", "citedby_count"]],
                    return_html=True
                )
                st.components.v1.html(walker_html)
            except Exception as e:
                st.error(f"Error generating PyGWalker visualization: {e}")


    def enrich_and_process_data(self):
        """
        Enrich the DataFrame with SNIP values, reformat columns, and prepare the data.
        """
        if self.df.empty:
            st.info("No data available to process. Please upload a file or execute a query.")
            return  # Early exit if DataFrame is empty

        with st.spinner("Retrieving SNIP values from Elsevier..."):
            self.df = DataProcessor.enrich_with_snip(self.df)

        desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
        other_columns = [col for col in self.df.columns if col not in desired_column_order]
        final_column_order = desired_column_order + other_columns
        self.df = self.df[final_column_order]

        # Format MonthYear and Year columns
        self.df["MonthYear"] = self.df["Month"].astype(str) + "-" + self.df["Year"].astype(str)
        self.df["MonthYear"] = self.df["MonthYear"].str.replace(r"\.0", "", regex=True)
        self.df["MonthYear"] = pd.to_datetime(self.df["MonthYear"], format='%m-%Y', errors='coerce')
        self.df["Year"] = pd.to_numeric(self.df["Year"], errors='coerce')
        self.df['Year'] = self.df['Year'].astype('category')

        # Filter data for the last 5 years
        self.df_last_5_years = self.df[self.df['Year'].astype(float) >= (self.current_year - 4)]

    def display_publications_with_snip(self):
        """
        Display the publications enriched with SNIP values in a structured manner.
        """
        if hasattr(self, 'df') and not self.df.empty:
            st.header("Publications with Impact Factor (SNIP)")
    
            # Retrieve SNIP values and enrich the DataFrame
            with st.spinner("Retrieving SNIP values from Elsevier..."):
                self.df = DataProcessor.enrich_with_snip(self.df)
    
            # Reorder columns based on preferences
            desired_column_order = ["SNIP", "title", "Year", "Month", "author_names"]
            other_columns = [col for col in self.df.columns if col not in desired_column_order]
            final_column_order = desired_column_order + other_columns
            self.df = self.df[final_column_order]
    
            # Format MonthYear and Year columns
            self.df["MonthYear"] = self.df["Month"].astype(str) + "-" + self.df["Year"].astype(str)
            self.df["MonthYear"] = self.df["MonthYear"].str.replace(r"\.0", "", regex=True)
            self.df["MonthYear"] = pd.to_datetime(self.df["MonthYear"], format='%m-%Y', errors='coerce')
            self.df["Year"] = pd.to_numeric(self.df["Year"], errors='coerce')
            self.df['Year'] = self.df['Year'].astype('category')
            
            self.df = self.df.sort_values(by="SNIP", ascending=False)
    
            # Display the enriched DataFrame
            st.write(self.df)
        else:
            st.warning("No publications to display. Please load or generate data first.")

    def run(self):
        st.title("Publication Metrics Dashboard")
        st.markdown("""
        This app allows you to explore publication metrics for the Division of Molecular and Translational BioMedicine.
        You can either upload a spreadsheet (e.g., a publication report), a word document with DOI list, **or** enter and execute a Scopus query.
        The app aggregates publications over time, enriches them with SNIP values, and allows for building plots to analyze publication stats.
        """)
    
        self.display_sidebar()
    
        # Reinitialize DataFrame from session state
        if "scopus_df" in st.session_state and not st.session_state["scopus_df"].empty:
            self.df = st.session_state["scopus_df"]
    
        if self.df.empty:
            st.info("Please upload a publication file or execute a Scopus query from the sidebar.")
        else:
            self.enrich_and_process_data()
            self.display_publications_with_snip()
            self.display_scopus_data()
            self.render_line_graph()
            self.render_violin_plot()
            self.render_coauthor_network()


    def display_sidebar(self):
        """
        Display the sidebar for uploading files or entering queries.
        """
        st.sidebar.header("API & Data Input Settings")
        data_source = st.sidebar.radio("Select Data Source", ["Scopus Query", "Upload Spreadsheet"])
        self.df = pd.DataFrame()  # Avoid reinitializing unless necessary
    
        if data_source == "Upload Spreadsheet":
            uploaded_file = st.sidebar.file_uploader("Upload Publications File (CSV, Excel, or DOCX)", type=["csv", "xls", "xlsx", "docx"])
            if uploaded_file:
                if hasattr(self, 'handle_uploaded_file'):
                    self.df = self.handle_uploaded_file(uploaded_file)
                    if self.df.empty:
                        st.warning("No valid data found in the uploaded file.")
                    else:
                        st.session_state["scopus_df"] = self.df  # Persist DataFrame to session state
                        st.success("File uploaded and processed successfully!")
                else:
                    st.error("The 'handle_uploaded_file' method is not defined in this class.")
    
        elif data_source == "Scopus Query":
            st.sidebar.subheader("Enter Search Parameters")
    
            with st.sidebar.expander("LLM Query Conversion", expanded=False):
                conversion_method = st.selectbox(
                    "Select Conversion Method",
                    ["Unformatted to Scopus Query", "PubMed to Scopus Query"]
                )
                st.text_area("Enter your query", height=100, key="input_query")
    
                if st.button("Convert Query", key="llm_query_convert"):
                    prompt_type = "pubmed" if conversion_method == "PubMed to Scopus Query" else "generic"
                    converted_query = QueryConverter.convert_query(
                        st.session_state.input_query, 
                        prompt_type, 
                        self.api_headers, 
                        self.openai_api_base
                    )
    
                    if converted_query:
                        st.session_state.scopus_query = converted_query
                    else:
                        st.error("Conversion failed. Possible reasons: malformed input or API issues.")
    
            st.text_area(
                "Enter Scopus Query Directly",
                value=st.session_state.get("scopus_query", ""),
                height=100,
                key="direct_query"
            )
    
            if st.sidebar.button("Execute Query", key="execute_scopus"):
                with st.spinner("Executing Scopus query..."):
                    self.df = DataProcessor.fetch_scopus_data(st.session_state.scopus_query)
                    if self.df.empty:
                        st.warning("No data found for the Scopus query. Please refine your query.")
                    else:
                        self.df = DataProcessor.process_data(self.df)
                        st.session_state.scopus_df = self.df  # Persist DataFrame to session state


    def render_line_graph(self):
        """
        Render a line graph showing monthly publication trends over the last 5 years.
        """
        if hasattr(self, 'df_last_5_years') and not self.df_last_5_years.empty:
            monthly_counts_last_5_years, _ = DataProcessor.aggregate_counts(self.df_last_5_years)
            grouped_counts_last_5_years = monthly_counts_last_5_years.copy()
            grouped_counts_last_5_years['Year'] = grouped_counts_last_5_years['Year'].astype(int)
            grouped_counts_last_5_years['Month'] = grouped_counts_last_5_years['Month'].astype(int)

            # Establish date range
            min_date = datetime.datetime(
                grouped_counts_last_5_years['Year'].min(),
                grouped_counts_last_5_years['Month'].min(),
                1
            )
            max_date = datetime.datetime(
                grouped_counts_last_5_years['Year'].max(),
                grouped_counts_last_5_years['Month'].max(),
                1
            )
            complete_date_range_last_5_years = pd.date_range(start=min_date, end=max_date, freq='MS')

            # Create a DataFrame for the complete date range
            date_range_df_last_5_years = pd.DataFrame({
                'YearMonth': complete_date_range_last_5_years.strftime('%Y-%m'),
                'Year': complete_date_range_last_5_years.year,
                'Month': complete_date_range_last_5_years.month
            })

            # Merge to fill missing months with 0 counts
            grouped_counts_last_5_years = pd.merge(
                date_range_df_last_5_years,
                grouped_counts_last_5_years,
                how='left',
                on=['Year', 'Month']
            )
            grouped_counts_last_5_years['Count'] = grouped_counts_last_5_years['Count'].fillna(0)
            grouped_counts_last_5_years['YearMonth'] = grouped_counts_last_5_years['YearMonth'].astype(str)

            # Filter for the last 5 years
            grouped_counts_last_5_years = grouped_counts_last_5_years[
                grouped_counts_last_5_years['Year'] >= (self.current_year - 4)
            ]

            # Line Graph for the Last 5 Years
            st.subheader("Publication Trends (Last 5 Years)")
            fig = px.line(
                grouped_counts_last_5_years,
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
                    text="Monthly Publication Trend (Last 5 Years",
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
        else:
            st.warning("No publication data available for rendering trends over the last 5 years.")
            
    def render_violin_plot(self):
        """
        Render a violin plot showing SNIP distribution by year for the last 5 years.
        """
        if hasattr(self, 'df_last_5_years') and not self.df_last_5_years.empty:
            st.header("Violin Plot of SNIP Distribution (Last 5 Years)")
    
            # Create violin plot
            fig = px.violin(
                self.df_last_5_years,
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
    
            # Display plot
            st.plotly_chart(fig, use_container_width=True, key="violin_plot_last_5_years")
        else:
            st.warning("No publication data available for rendering the violin plot.")
            
    def filter_network(self, network, min_collaborations=4):
        """
        Filter the network to include only edges with weight >= min_collaborations.
    
        Args:
            network (networkx.Graph): The original coauthor network graph.
            min_collaborations (int): Minimum number of collaborations to keep an edge.
    
        Returns:
            networkx.Graph: A filtered graph with edges meeting the minimum collaboration criteria.
        """
        filtered_network = nx.Graph()
        for u, v, data in network.edges(data=True):
            if data.get("weight", 0) >= min_collaborations:
                filtered_network.add_edge(u, v, weight=data["weight"])
        return filtered_network
            
    def render_coauthor_network(self):
        """
        Render a coauthor network visualization for the last 5 years.
        """
        if hasattr(self, 'df_last_5_years') and 'author_names' in self.df_last_5_years.columns:
            st.write("### Co-Author Network Visualization (Last 5 Years)")
    
            # Build the co-author network
            coauthor_network = NetworkBuilder.build_coauthor_network(self.df_last_5_years)
    
            # Filter by minimum collaborations dynamically using a slider
            min_collaborations = st.slider("Minimum Collaborations to Display", 1, 10, 4)
            filtered_coauthor_network = self.filter_network(coauthor_network, min_collaborations=min_collaborations)
    
            # Visualize the graph
            fig, ax = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(filtered_coauthor_network, seed=42, k=1.2, iterations=100)  # Adjusted `k` and iterations
    
            # Draw nodes with a standard size
            nx.draw_networkx_nodes(
                filtered_coauthor_network, pos, node_size=200, node_color="green", alpha=0.8
            )
    
            # Draw edges with a standard width
            nx.draw_networkx_edges(
                filtered_coauthor_network, pos, width=1.5, alpha=0.7, edge_color="gray"
            )
    
            # Add node labels
            nx.draw_networkx_labels(filtered_coauthor_network, pos, font_size=10, font_color="black")
    
            plt.title("Co-Author Network (Last 5 Years)", fontsize=14)
            plt.axis("off")  # Remove axes
            st.pyplot(fig)
        else:
            st.warning("No author data available for building the coauthor network. Please check the input data.")

# =============================================================================
# Run the App
# =============================================================================
if __name__ == "__main__":
    app = AdvancedMetricsApp()
    app.run()
