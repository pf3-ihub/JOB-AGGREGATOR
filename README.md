# Job Market Analysis Dashboard

## Overview

This interactive Streamlit dashboard analyzes job posting data to provide comprehensive insights into job market trends, required skills, qualifications, and compensation. It helps recruiters, job seekers, and HR professionals understand the current job market landscape through data visualization and detailed analysis.

## Features

- **Job Categories Analysis**: Visualize distribution and trends of job categories over time
- **Skills Analysis**: Identify in-demand technical and soft skills across different job roles
- **Experience & Education Requirements**: Analyze required experience levels and educational qualifications
- **Compensation & Benefits**: Explore salary ranges and benefits offered across job categories
- **Company Comparison**: Compare job offerings across multiple companies
- **Detailed Job Listings**: Search and filter job listings with comprehensive details
- **Export Functionality**: Download processed data and filtered results

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/job-market-analysis-dashboard.git
   cd job-market-analysis-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run final.py
   ```

2. Open your web browser and go to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your job data CSV file(s) using the sidebar uploader

### Data Format

The dashboard expects CSV files with the following columns:
- `Job Title`: Title of the job posting
- `Company`: Company name
- `Location`: Job location
- `Date Posted`: When the job was posted
- `Description`: Full job description
- `Responsibilities`: Job responsibilities
- `Minimum Qualifications`: Required qualifications
- `Preferred Qualifications`: Preferred qualifications

Additional columns are optional and will be utilized if present.

### Example CSV Format

```
Job Title,Company,Location,Date Posted,Description,Responsibilities,Minimum Qualifications,Preferred Qualifications
Software Engineer,TechCorp,San Francisco CA,2023-01-15,"Job description text...","Responsibilities text...","Required qualifications...","Preferred qualifications..."
```

## Data Processing

The dashboard performs the following data processing:

1. **Data Cleaning**: Standardizes job titles, handles missing values
2. **Feature Extraction**:
   - Categorizes jobs into industry sectors
   - Extracts experience and education requirements
   - Identifies required skills from job descriptions
   - Determines work mode (remote/hybrid/onsite)
   - Extracts salary information and benefits

## Dashboard Navigation

The dashboard is organized into six main tabs:

1. **Overview**: Key metrics and general distribution of jobs
2. **Skills Analysis**: In-depth analysis of required technical and soft skills
3. **Job Categories**: Detailed breakdown of job categories and subcategories
4. **Company Comparison**: Compare job offerings across companies
5. **Compensation & Benefits**: Analyze salary ranges and benefits
6. **Job Details**: Search and explore detailed job listings

## Troubleshooting

- **File Upload Issues**: Ensure your CSV files are properly formatted with UTF-8 encoding
- **Missing Data**: The dashboard will handle missing columns, but having complete data will provide better insights
- **Performance Issues**: For large datasets, consider filtering by date range or company to improve performance

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Analysis tools from [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Word cloud generation using [WordCloud](https://github.com/amueller/word_cloud)