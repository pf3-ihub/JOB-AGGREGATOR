import streamlit as st
st.set_page_config(page_title="Job Market Analysis Hub", layout="wide", initial_sidebar_state="expanded")
import pandas as pd
import os
import final  # Import your existing functions

# Set page config must be the first Streamlit command
# st.set_page_config(page_title="Job Market Analysis Hub", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .company-name {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    .company-stats {
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
    }
    .header-style {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 10px;
        border-bottom: 3px solid #4c9be8;
    }
    .company-tile {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 15px;
    }
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define the data directory - adjust this to your needs
DATA_DIR = "data"

def load_company_data(data_path):
    """Load and process the company data from a CSV file"""
    try:
        df = pd.read_csv(data_path)
        # Add company name to dataframe based on filename
        company_name = os.path.basename(data_path).split("_")[0].capitalize()
        df['Company'] = company_name
        # Process the data
        return final.load_and_clean_data_from_df(df)
    except Exception as e:
        st.error(f"Error loading data for {data_path}: {str(e)}")
        return None

def display_company_tile(col, company_name, logo_path, job_count):
    """Display company logo, name, job count and view button"""
    with col:
        st.markdown('<div class="company-tile">', unsafe_allow_html=True)
        # Display logo
        if os.path.exists(logo_path):
            # Display the logo directly
            st.image(logo_path, width=100)
        else:
            # Simple placeholder if no logo exists
            st.markdown(f'<div style="width:80px;height:80px;background-color:#f0f2f6;display:flex;align-items:center;justify-content:center;border-radius:10px;"><h2>{company_name[0]}</h2></div>', unsafe_allow_html=True)
        
        # Add company name and job count
        st.markdown(f'<div class="company-name">{company_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="company-stats">{job_count} Jobs</div>', unsafe_allow_html=True)
        
        # Add View button
        if st.button("View", key=f"btn_{company_name}"):
            st.session_state.selected_company = company_name
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main function
def main():
    # Initialize session state for storing selected company
    if 'selected_company' not in st.session_state:
        st.session_state.selected_company = None
    
    # Main page description
    st.markdown('<div class="header-style">Job Market Analysis Hub</div>', unsafe_allow_html=True)
    st.write("Explore job market data by company or navigate to the Trends and Predictions pages")
    
    # If a company is selected, show the dashboard for that company
    if st.session_state.selected_company:
        company_name = st.session_state.selected_company
        
        # Add a back button at the top
        if st.button("← Back to Companies"):
            st.session_state.selected_company = None
            st.rerun()
        
        # Find the CSV file for this company
        company_file = None
        for file in os.listdir(DATA_DIR):
            if file.lower().startswith(company_name.lower()) and file.endswith('.csv'):
                company_file = os.path.join(DATA_DIR, file)
                break
        
        if company_file:
            # Load and display the dashboard
            df = load_company_data(company_file)
            if df is not None:
                final.create_dashboard(df)
            else:
                st.error(f"Could not load data for {company_name}")
        else:
            st.error(f"No data file found for {company_name}")
    else:
        # Show the company selection page
        # Get available companies from the data directory
        companies = []
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                if file.endswith('.csv'):
                    # Extract company name from filename (assuming format like "company_jobs.csv")
                    company_name = file.split('_')[0].capitalize()
                    
                    # Count jobs in the CSV
                    try:
                        df = pd.read_csv(os.path.join(DATA_DIR, file))
                        job_count = len(df)
                    except:
                        job_count = 0
                    
                    # Look for company logo
                    logo_path = os.path.join('logos', f"{company_name.lower()}_logo.png")
                    
                    companies.append({
                        "name": company_name,
                        "logo_path": logo_path,
                        "job_count": job_count,
                        "data_path": os.path.join(DATA_DIR, file)
                    })
        
        if not companies:
            st.warning("No company data found. Please add CSV files to the 'data' directory.")
            
            # Add instructions for data structure
            st.markdown("### Data Directory Structure")
            st.markdown("""
            Create the following directory structure:
            ```
            your_app/
            ├── data/
            │   ├── google_jobs.csv
            │   ├── amazon_jobs.csv
            │   └── ...
            ├── logos/
            │   ├── google_logo.png
            │   ├── amazon_logo.png
            │   └── ...
            ├── final.py
            ├── index.py
            └── pages/
                ├── 1_Trends.py
                └── 2_Predictions.py
            ```
            
            Your CSV files should contain the columns used in your dashboard.
            """)
        else:
            # Navigation instructions
            st.info("👈 Use the sidebar to navigate between Company Analysis, Market Trends, and Future Predictions")
            
            # Create a container for better centering
            with st.container():
                st.markdown('<div class="main-content">', unsafe_allow_html=True)
                
                # Create a grid of company tiles
                cols = st.columns(3)  # 3 columns grid
                for i, company in enumerate(companies):
                    display_company_tile(
                        cols[i % 3],
                        company["name"], 
                        company["logo_path"], 
                        company["job_count"]
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()