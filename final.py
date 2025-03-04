import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from urllib.parse import quote
import json

# Set page config
st.set_page_config(page_title="Job Market Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Add custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4c9be8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-label {
        font-weight: bold;
        color: #555;
    }
    .header-style {
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
        border-bottom: 2px solid #ddd;
        padding-bottom: 10px;
    }
    .subheader-style {
        font-size: 22px;
        font-weight: bold;
        color: #555;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 3px solid #4c9be8;
        padding-left: 10px;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    /* Apply hover effect on table rows */
    tbody tr:hover {
        background-color: rgba(66, 133, 244, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Error handling decorators
def handle_exceptions(func):
    """Decorator to handle exceptions in function execution"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.exception(e)
            return None
    return wrapper

# Function to create downloadable link for dataframe
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def clean_job_title(title):
    """Clean job titles to standardize them"""
    if not isinstance(title, str):
        return "Unknown"
    
    # Remove special characters and standardize
    title = re.sub(r'[^\w\s-]', '', title).strip()
    
    # Handle common prefixes (can be customized for different companies)
    common_prefixes = ["IN ", "IN-", "US ", "US-", "UK ", "UK-"]
    for prefix in common_prefixes:
        if title.startswith(prefix):
            title = title.replace(prefix, "")
    
    return title

def extract_job_category(title):
    """Extract main job category from title"""
    if not isinstance(title, str):
        return "Other"
    
    title = title.lower()
    
    # Define job categories with related keywords - expanded for more depth
    categories = {
        "Software Engineering": ["software engineer", "swe", "developer", "programming", "coder", "full stack", "frontend", "backend", "mobile dev", "web developer"],
        "Hardware Engineering": ["hardware engineer", "hardware design", "electronics", "electrical engineer", "circuit", "pcb"],
        "Data Science": ["data scientist", "data analyst", "machine learning", "ml engineer", "ai", "analytics", "statistician", "big data"],
        "DevOps": ["devops", "site reliability", "sre", "infrastructure", "platform engineer", "cloud engineer", "operations engineer"],
        "QA & Testing": ["qa", "test", "quality assurance", "quality engineer", "tester", "test automation"],
        "Product Management": ["product manager", "product owner", "program manager", "project manager", "technical program manager"],
        "UX/UI Design": ["ux", "ui", "user experience", "user interface", "designer", "design", "creative"],
        "Business & Sales": ["business", "sales", "account", "client", "customer", "marketing", "growth", "advocacy"],
        "Support & Operations": ["support", "specialist", "operations", "expert", "technician", "service", "customer support"],
        "Management": ["manager", "management", "lead", "director", "head of", "chief", "vp"],
        "Security": ["security", "infosec", "cyber", "penetration tester", "ethical hacker", "compliance", "risk"],
    }
    
    for category, keywords in categories.items():
        if any(keyword in title.lower() for keyword in keywords):
            return category
    
    return "Other"

def extract_subcategory(title, category):
    """Extract more specific job subcategory based on the main category"""
    if not isinstance(title, str) or category == "Other":
        return "General"
    
    title = title.lower()
    
    # Define subcategories for each main category
    subcategories = {
        "Software Engineering": {
            "Frontend": ["frontend", "front end", "front-end", "ui", "react", "angular", "vue", "javascript", "web developer"],
            "Backend": ["backend", "back end", "back-end", "server", "api", "database", "java", "python", "node", "php", "ruby", "go"],
            "Full Stack": ["full stack", "full-stack", "fullstack"],
            "Mobile": ["ios", "android", "mobile", "app developer", "swift", "kotlin", "react native", "flutter"],
            "Embedded": ["embedded", "firmware", "low level", "kernel", "driver", "rtos"],
            "Cloud": ["cloud", "aws", "azure", "gcp", "serverless"],
        },
        "Data Science": {
            "Machine Learning": ["machine learning", "ml", "deep learning", "neural", "nlp", "computer vision", "ai"],
            "Data Analysis": ["data analyst", "analytics", "bi ", "business intelligence", "tableau", "power bi", "looker"],
            "Data Engineering": ["data engineer", "etl", "data pipeline", "hadoop", "spark", "kafka"],
        },
        "DevOps": {
            "SRE": ["site reliability", "sre", "reliability"],
            "Platform Engineering": ["platform", "infrastructure"],
            "Cloud Operations": ["cloud", "aws", "azure", "gcp", "devops"],
        },
        "QA & Testing": {
            "Manual Testing": ["manual test", "manual qa"],
            "Automation Testing": ["automation", "automated", "selenium", "cypress", "test automation"],
            "Security Testing": ["security test", "penetration", "pen test"]
        },
        "Security": {
            "Application Security": ["application security", "appsec", "secure coding"],
            "Network Security": ["network security", "firewall", "vpn"],
            "Security Operations": ["soc", "security operations", "incident response"]
        },
        "Product Management": {
            "Technical PM": ["technical product", "tpm", "technical program"],
            "Growth PM": ["growth", "user acquisition"],
            "Core PM": ["core product"]
        },
        "UX/UI Design": {
            "UX Research": ["ux research", "user research", "usability"],
            "UI Design": ["ui design", "interface design", "visual design"],
            "Interaction Design": ["interaction", "ixd", "motion"]
        }
    }
    
    if category in subcategories:
        for subcategory, keywords in subcategories[category].items():
            if any(keyword in title for keyword in keywords):
                return subcategory
    
    return "General"

def extract_experience_requirement(text):
    """Extract years of experience requirement from text"""
    if not isinstance(text, str):
        return None
    
    # Look for patterns like "5+ years", "5-7 years", "10+ years"
    patterns = [
        r'(\d+)\+\s*years?',                   # 5+ years
        r'(\d+)[-–]\d+\s*years?',              # 5-7 years (capture lower bound)
        r'(\d+)\s*years?\s*of\s*experience',   # 5 years of experience
        r'(\d+)\s*\+\s*years?\s*of\s*experience',  # 5 + years of experience
        r'minimum\s*of\s*(\d+)\s*years?',      # minimum of 5 years
        r'at\s*least\s*(\d+)\s*years?',        # at least 5 years
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    
    return None

def extract_education_requirements(text):
    """Extract education requirements from text"""
    if not isinstance(text, str):
        return "Not Specified"
    
    text = text.lower()
    education_patterns = {
        "PhD": ["phd", "ph.d", "doctorate", "doctoral"],
        "Master's": ["master", "ms ", "m.s", "graduate", "master's", "mba"],
        "Bachelor's": ["bachelor", "bs ", "b.s", "undergraduate", "be/btech", "be / btech", "bachelor's"],
        "Associate's": ["associate", "a.s", "a.a"]
    }
    
    # Check PhD first, then Master's, then Bachelor's (higher degrees take precedence)
    for level, patterns in education_patterns.items():
        if any(pattern in text for pattern in patterns):
            return level
    
    return "Not Specified"

def extract_skills(text):
    """Extract key skills from job description and qualifications"""
    if not isinstance(text, str):
        return []
    
    # Expanded list of technical skills to look for
    skill_patterns = [
        # Programming Languages
        r'\bjava\b', r'\bpython\b', r'\bc\+\+\b', r'\bc#\b', r'\bruby\b', r'\brust\b', 
        r'\bgo\b', r'\bgolang\b', r'\bscala\b', r'\bphp\b', r'\bperl\b', r'\bswift\b',
        r'\bkotlin\b', r'\btypescript\b', r'\bhaskell\b', r'\berlang\b', r'\belixir\b',
        r'\bluа\b', r'\bobjective-c\b', r'\bgroovy\b', r'\bdart\b', r'\bfortran\b',
        
        # Front-end
        r'\bjavascript\b', r'\bjs\b', r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bsvelte\b',
        r'\bhtml5?\b', r'\bcss3?\b', r'\bsass\b', r'\bless\b', r'\bwebpack\b', r'\bvite\b',
        r'\bgulp\b', r'\bember\b', r'\bjquery\b', r'\bdom\b', r'\bbootstrap\b', r'\btailwind\b',
        
        # Back-end
        r'\bnode\.?js\b', r'\bexpress\b', r'\bdjango\b', r'\bflask\b', r'\brails\b',
        r'\bspring\b', r'\bhibernate\b', r'\blaravel\b', r'\b\.net\b', r'\basp\.net\b',
        
        # Cloud
        r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bcloud\b', r'\bgoogle cloud\b', r'\bs3\b',
        r'\blambda\b', r'\bec2\b', r'\beks\b', r'\baks\b', r'\bgke\b', r'\brds\b',
        r'\bdynamodb\b', r'\bcosmos db\b', r'\bfirebase\b',
        
        # DevOps/Infra
        r'\bdocker\b', r'\bkubernetes\b', r'\bk8s\b', r'\bterraform\b', r'\bansible\b',
        r'\bchef\b', r'\bpuppet\b', r'\bcircleci\b', r'\bjenkins\b', r'\bgithub actions\b',
        r'\btravic ci\b', r'\bargo\b', r'\bflux\b', r'\bhelm\b', r'\bvagrant\b',
        
        # Databases
        r'\bsql\b', r'\bnosql\b', r'\bmongodb\b', r'\bcassandra\b', r'\belasticsearch\b',
        r'\bmysql\b', r'\bpostgres\b', r'\boracle\b', r'\bsqlite\b', r'\bredis\b',
        r'\bcouchbase\b', r'\bneo4j\b', r'\bgraphql\b', r'\bbigtable\b', r'\binfluxdb\b',
        
        # OS and Platform
        r'\blinux\b', r'\bunix\b', r'\bwindows\b', r'\bmac\b', r'\bios\b', r'\bandroid\b',
        r'\bubuntu\b', r'\bcentos\b', r'\bredhat\b', r'\bdebian\b', r'\bfedora\b',
        
        # Data Science & AI
        r'\bml\b', r'\bmachine learning\b', r'\bai\b', r'\bartificial intelligence\b', 
        r'\bdata science\b', r'\bpandas\b', r'\bnumpy\b', r'\bscikit-learn\b', r'\btensorflow\b',
        r'\bpytorch\b', r'\bkeras\b', r'\bcv\b', r'\bcomputer vision\b', r'\bnlp\b',
        r'\bnatural language processing\b', r'\br language\b', r'\bjupyter\b', r'\bscipy\b',
        r'\bmxnet\b', r'\bcaffe\b', r'\btheano\b', r'\btorch\b',
        
        # Big Data
        r'\bspark\b', r'\bhadoop\b', r'\bbig data\b', r'\betl\b', r'\bhive\b', r'\bpig\b',
        r'\bkafka\b', r'\bflume\b', r'\bzookeeper\b', r'\bflink\b', r'\bavro\b', r'\bparquet\b',
        
        # Testing
        r'\btest automation\b', r'\bselenium\b', r'\bjunit\b', r'\bmockito\b', r'\bject\b',
        r'\btest-driven\b', r'\btdd\b', r'\bbdd\b', r'\bcucumber\b', r'\bjest\b', r'\bmocha\b',
        r'\bjavascript testing\b', r'\bqunit\b', r'\bcypress\b', r'\bprotractor\b', r'\bwebdriver\b',
        
        # API
        r'\brest\b', r'\bapi\b', r'\bsoap\b', r'\bopenapi\b', r'\bswagger\b', r'\bgraphql\b',
        r'\bhttp\b', r'\bjson\b', r'\bxml\b', r'\bprotobuf\b', r'\bgrpc\b', r'\bwebhooks\b',
        
        # Methodologies
        r'\bscrum\b', r'\bagile\b', r'\bkanban\b', r'\bwaterfall\b', r'\blean\b', r'\bsafe\b',
        r'\bdevops\b', r'\bci/cd\b', r'\bcontinuous integration\b', r'\bcontinuous delivery\b',
        
        # Security
        r'\bsecurity\b', r'\bcryptography\b', r'\bpki\b', r'\bssl\b', r'\bnetworking\b', r'\btcp/ip\b',
        r'\bowasp\b', r'\bpenetration testing\b', r'\bpen testing\b', r'\binfosec\b', r'\bsoc\b',
        
        # Embedded
        r'\bembedded\b', r'\bfirmware\b', r'\brtos\b', r'\braspberry pi\b', r'\barduino\b',
        r'\bmicrocontroller\b', r'\bvhdl\b', r'\bverilog\b', r'\barmc?\b', r'\bdriver\b'
    ]
    
    found_skills = []
    for pattern in skill_patterns:
        if re.search(pattern, text.lower()):
            # Clean up the skill name
            skill = pattern.replace(r'\b', '').replace('\\', '').replace('?', '')
            found_skills.append(skill)
    
    return found_skills

def extract_key_terms(text, term_list):
    """Extract occurrences of key terms from text"""
    if not isinstance(text, str):
        return []
    
    found_terms = []
    for term in term_list:
        if re.search(r'\b' + re.escape(term.lower()) + r'\b', text.lower()):
            found_terms.append(term)
    
    return found_terms

def extract_salary_range(text):
    """Extract salary information from job posting"""
    if not isinstance(text, str):
        return None, None
    
    # Look for patterns like "$50,000-$70,000", "$50k-$70k", "$50,000 per year"
    salary_patterns = [
        r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # $50,000-$70,000
        r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*to\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # $50,000 to $70,000
        r'\$(\d{1,2})k\s*-\s*\$(\d{1,2})k',  # $50k-$70k
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|GBP)',  # 50,000-70,000 USD
        r'salary range:?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # salary range: $50,000-$70,000
    ]
    
    for pattern in salary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Convert string numbers to floats, removing commas
            try:
                min_salary = float(match.group(1).replace(',', '').replace('k', '000'))
                max_salary = float(match.group(2).replace(',', '').replace('k', '000'))
                return min_salary, max_salary
            except (ValueError, IndexError):
                continue
    
    # Look for single salary values like "$50,000 per year"
    single_salary_patterns = [
        r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:per|a|/)\s*(?:year|annum|yr)',  # $50,000 per year
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|GBP)\s*(?:per|a|/)\s*(?:year|annum|yr)',  # 50,000 USD per year
    ]
    
    for pattern in single_salary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                salary = float(match.group(1).replace(',', ''))
                return salary, salary  # Same value for min and max
            except (ValueError, IndexError):
                continue
    
    return None, None

def extract_work_mode(text):
    """Extract work mode (Remote/Onsite/Hybrid) from text"""
    if not isinstance(text, str):
        return "Not Specified"
    
    text = text.lower()
    
    if any(term in text for term in ["remote work", "work from home", "wfh", "fully remote", "100% remote", "remote position", "remote job"]):
        return "Remote"
    elif any(term in text for term in ["hybrid", "flexible work", "partially remote", "remote flexible", "flexible remote", "office flexible"]):
        return "Hybrid"
    elif any(term in text for term in ["onsite", "in office", "in-office", "on location", "on-site", "workplace", "accommodation", "on site", "in person"]):
        return "Onsite"
    else:
        return "Not Specified"

def extract_job_benefits(text):
    """Extract common job benefits mentioned in the posting"""
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    
    benefits = {
        "Health Insurance": ["health insurance", "medical insurance", "healthcare", "health benefits"],
        "Dental Insurance": ["dental insurance", "dental benefits", "dental coverage"],
        "Vision Insurance": ["vision insurance", "vision benefits", "eye care"],
        "401(k)": ["401k", "401(k)", "retirement plan", "retirement benefits"],
        "PTO": ["paid time off", "pto", "vacation time", "vacation days", "annual leave"],
        "Remote Work": ["remote work", "work from home", "wfh", "remote position"],
        "Flexible Hours": ["flexible hours", "flexible schedule", "flex time", "flextime"],
        "Stock Options": ["stock options", "equity", "rsus", "stock grants"],
        "Parental Leave": ["parental leave", "maternity leave", "paternity leave", "family leave"],
        "Professional Development": ["professional development", "training budget", "learning budget", "education reimbursement"],
        "Gym Membership": ["gym membership", "fitness", "wellness program"],
        "Relocation Assistance": ["relocation", "relocation assistance", "moving expenses"],
        "Bonus": ["bonus", "performance bonus", "annual bonus", "sign-on bonus", "signing bonus"],
        "Free Food": ["free food", "catered meals", "lunch provided", "snacks provided"]
    }
    
    found_benefits = []
    for benefit, keywords in benefits.items():
        if any(keyword in text for keyword in keywords):
            found_benefits.append(benefit)
    
    return found_benefits

def experience_to_range(years):
    """Convert years of experience to range categories for analysis"""
    if pd.isna(years):
        return "Not Specified"
    
    if years <= 1:
        return "Entry Level (0-1 years)"
    elif years <= 3:
        return "Junior (1-3 years)"
    elif years <= 5:
        return "Mid-Level (3-5 years)"
    elif years <= 8:
        return "Senior (5-8 years)"
    else:
        return "Expert (8+ years)"

@handle_exceptions
def load_and_clean_data(file_path):
    """Load and clean the dataset"""
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist, create them if not
        required_columns = [
            "Job Title", "Location", "Date Posted", "Description", 
            "Responsibilities", "Minimum Qualifications", "Preferred Qualifications"
        ]
        
        # Fix column names if they're slightly different (e.g., Responsiblities instead of Responsibilities)
        rename_map = {}
        for col in df.columns:
            # Check for misspelled columns
            if col == "Responsiblities":
                rename_map[col] = "Responsibilities"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Add missing columns with empty values
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Check if "Company" column exists, if not, try to extract from filename or add placeholder
        if "Company" not in df.columns:
            company_name = get_company_name_from_path(file_path)
            df["Company"] = company_name
        
        # Fill missing values
        df['Location'] = df['Location'].fillna('Unknown')
        df['Job Title'] = df['Job Title'].fillna('Unknown')
        df['Description'] = df['Description'].fillna('')
        df['Responsibilities'] = df['Responsibilities'].fillna('')
        df['Minimum Qualifications'] = df['Minimum Qualifications'].fillna('')
        df['Preferred Qualifications'] = df['Preferred Qualifications'].fillna('')
        
        # Process location - Extract city and country if available
        df['City'] = df['Location'].apply(
            lambda x: x.split(",")[0].strip() if isinstance(x, str) and "," in x else x
        )
        
        # Extract country from location if available
        df['Country'] = df['Location'].apply(
            lambda x: x.split(",")[-1].strip() if isinstance(x, str) and "," in x else 'Unknown'
        )
        
        # Clean and standardize job titles
        df['Clean_Job_Title'] = df['Job Title'].apply(clean_job_title)
        
        # Extract job categories
        df['Job_Category'] = df['Job Title'].apply(extract_job_category)
        
        # Extract job subcategories for deeper analysis
        df['Job_Subcategory'] = df.apply(
            lambda row: extract_subcategory(row['Job Title'], row['Job_Category']), 
            axis=1
        )
        
        # Convert Date Posted to datetime
        df['Date Posted'] = pd.to_datetime(df['Date Posted'], errors='coerce')
        
        # Set current date for rows with missing date
        current_date = datetime.now()
        df['Date Posted'] = df['Date Posted'].fillna(current_date)
        
        # Create full text field for comprehensive text analysis
        df['Full_Text'] = df.apply(
            lambda row: ' '.join([
                str(row['Job Title']), 
                str(row['Description']), 
                str(row['Responsibilities']), 
                str(row['Minimum Qualifications']), 
                str(row['Preferred Qualifications'])
            ]),
            axis=1
        )
        
        # Extract years of experience required
        df['Experience_Required'] = df.apply(
            lambda row: extract_experience_requirement(str(row['Minimum Qualifications'])), 
            axis=1
        )
        
        # Group experience into ranges for easier analysis
        df['Experience_Range'] = df['Experience_Required'].apply(lambda x: experience_to_range(x))
        
        # Extract education requirements
        df['Education_Required'] = df.apply(
            lambda row: extract_education_requirements(
                str(row['Minimum Qualifications']) + " " + str(row['Preferred Qualifications'])
            ),
            axis=1
        )
        
        # Extract skills from job descriptions and qualifications
        df['Skills'] = df.apply(
            lambda row: extract_skills(row['Full_Text']),
            axis=1
        )
        
        # Extract soft skills and competencies
        soft_skills = [
            'communication', 'teamwork', 'leadership', 'problem solving', 'analytical', 
            'creativity', 'adaptability', 'time management', 'critical thinking', 'collaboration',
            'interpersonal', 'presentation', 'negotiation', 'conflict resolution', 'customer service',
            'emotional intelligence', 'attention to detail', 'organization', 'prioritization',
            'decision making', 'mentoring', 'coaching', 'strategic thinking', 'initiative'
        ]
        
        df['Soft_Skills'] = df.apply(
            lambda row: extract_key_terms(row['Full_Text'], soft_skills),
            axis=1
        )
        
        # Create Remote/Onsite/Hybrid flag
        df['Work_Mode'] = df.apply(
            lambda row: extract_work_mode(row['Full_Text']),
            axis=1
        )
        
        # Extract potential salary information
        df['Min_Salary'], df['Max_Salary'] = zip(*df['Full_Text'].apply(extract_salary_range))
        
        # Calculate average salary when both min and max are available
        df['Avg_Salary'] = df.apply(
            lambda row: (row['Min_Salary'] + row['Max_Salary']) / 2 if pd.notna(row['Min_Salary']) and pd.notna(row['Max_Salary']) else None,
            axis=1
        )
        
        # Extract benefits
        df['Benefits'] = df['Full_Text'].apply(extract_job_benefits)
        
        # Add data quality indicators
        df['Data_Completeness'] = df.apply(
            lambda row: calculate_completeness(row),
            axis=1
        )
        
        return df
    except:
        print ("Error in loading data")

def calculate_completeness(row):
    """Calculate completeness score for a job posting"""
    required_fields = ['Job Title', 'Location', 'Description', 'Minimum Qualifications']
    optional_fields = ['Responsibilities', 'Preferred Qualifications', 'Date Posted']
    
    # Check required fields
    required_score = sum(1 for field in required_fields if pd.notna(row[field]) and str(row[field]).strip() != '') / len(required_fields)
    
    # Check optional fields
    optional_score = sum(1 for field in optional_fields if pd.notna(row[field]) and str(row[field]).strip() != '') / len(optional_fields)
    
    # Weighted score (required fields are more important)
    return (required_score * 0.7) + (optional_score * 0.3)

def get_company_name_from_path(file_path):
    """Extract company name from file path or name if available"""
    try:
        # Try to get filename from the path/upload object
        if hasattr(file_path, 'name'):
            filename = file_path.name
        else:
            filename = str(file_path).split('/')[-1]
        
        # Extract company name from filename (assuming format like "company_jobs.csv")
        company_name = filename.split('_')[0].split('.')[0].title()
        
        # Clean up common suffixes
        company_name = company_name.replace("Test", "").replace("Data", "").strip()
        
        return company_name if company_name else "Unknown Company"
    except:
        return "Unknown Company"

@handle_exceptions
def create_dashboard(df):
    """Create the main dashboard"""
    # Get company name for title
    company_name = df['Company'].iloc[0] if 'Company' in df.columns else "Multiple Companies"
    
    st.markdown(f'<div class="header-style">{company_name} Jobs Analysis Dashboard</div>', unsafe_allow_html=True)
    st.write("Interactive analysis of job openings - find insights about skills, trends, and requirements")
    
    # Data quality warning if needed
    completeness_avg = df['Data_Completeness'].mean() if 'Data_Completeness' in df.columns else 0
    if completeness_avg < 0.7:
        st.warning("⚠️ Data quality warning: Some job listings may have incomplete information. Analysis results may be limited.")
    
    # Download processed data option
    st.sidebar.markdown("### Download Processed Data")
    st.sidebar.markdown(get_table_download_link(df, f"{company_name.lower().replace(' ', '_')}_processed.csv", "📥 Download Processed Data"), unsafe_allow_html=True)
    
    # Company selector if multiple companies
    if 'Company' in df.columns and df['Company'].nunique() > 1:
        companies = ['All Companies'] + sorted(df['Company'].unique().tolist())
        selected_company = st.selectbox("Select Company", companies)
        
        if selected_company != 'All Companies':
            df_filtered = df[df['Company'] == selected_company]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Time period filter
    if 'Date Posted' in df.columns:
        st.sidebar.markdown("### Time Period Filter")
        min_date = df['Date Posted'].min().date()
        max_date = df['Date Posted'].max().date()
        
        date_range = st.sidebar.date_input(
            "Filter by date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[(df_filtered['Date Posted'].dt.date >= start_date) & 
                                      (df_filtered['Date Posted'].dt.date <= end_date)]
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🧠 Skills Analysis", 
        "📂 Job Categories", 
        "🏢 Company Comparison", 
        "💰 Compensation & Benefits",
        "📋 Job Details"
    ])
    
    with tab1:
        create_overview_tab(df_filtered)
    
    with tab2:
        create_skills_analysis_tab(df_filtered)
    
    with tab3:
        create_job_categories_tab(df_filtered)
    
    with tab4:
        create_company_comparison_tab(df)
    
    with tab5:
        create_compensation_tab(df_filtered)
        
    with tab6:
        create_job_details_tab(df_filtered)

@handle_exceptions
def create_overview_tab(df):
    """Create content for Overview tab"""
    # Key Metrics
    st.markdown('<div class="subheader-style">Key Metrics</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Jobs", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Job Categories", df['Job_Category'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_exp = df['Experience_Required'].dropna().mean()
        st.metric("Avg Experience Req.", f"{avg_exp:.1f} yrs" if not pd.isna(avg_exp) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        if 'Date Posted' in df.columns and not df['Date Posted'].isna().all():
            latest_date = df['Date Posted'].max().strftime('%d-%b-%Y')
        else:
            latest_date = "N/A"
        st.metric("Latest Job Posted", latest_date)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Job Categories Distribution
    cat_cols = st.columns([3, 2])
    
    with cat_cols[0]:
        st.markdown('<div class="subheader-style">Job Categories Distribution</div>', unsafe_allow_html=True)
        
        if df['Job_Category'].nunique() > 0:
            job_cat_counts = df['Job_Category'].value_counts()
            
            # If too many categories, limit to top categories and group others
            if len(job_cat_counts) > 8:
                top_categories = job_cat_counts.head(7)
                other_count = job_cat_counts[7:].sum()
                
                # Create new Series with top categories and "Other"
                job_cat_counts = pd.concat([top_categories, pd.Series({'Other': other_count})])
            
            fig_job_cat = px.pie(
                names=job_cat_counts.index, 
                values=job_cat_counts.values,
                title="Distribution of Job Categories",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig_job_cat.update_traces(textposition='inside', textinfo='percent+label')
            fig_job_cat.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=50, b=50, l=10, r=10)
            )
            st.plotly_chart(fig_job_cat, use_container_width=True)
        else:
            st.info("No job category data available")
    
    with cat_cols[1]:
        # Work Mode Distribution
        st.markdown('<div class="subheader-style">Work Mode Distribution</div>', unsafe_allow_html=True)
        
        if df['Work_Mode'].nunique() > 0:
            work_mode_counts = df['Work_Mode'].value_counts()
            
            fig_work_mode = px.pie(
                names=work_mode_counts.index,
                values=work_mode_counts.values,
                title="Remote vs. Onsite Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_work_mode.update_traces(textposition='inside', textinfo='percent+label')
            fig_work_mode.update_layout(
                margin=dict(t=50, b=0, l=10, r=10)
            )
            st.plotly_chart(fig_work_mode, use_container_width=True)
        else:
            st.info("No work mode data available")
    
    # Job Posting Trends
    st.markdown('<div class="subheader-style">Job Posting Trends by Category</div>', unsafe_allow_html=True)
    
    # Filter out rows with NaT dates
    df_date = df.dropna(subset=['Date Posted']).copy()
    
    if not df_date.empty:
        # Add month column for trend analysis
        df_date['Month'] = df_date['Date Posted'].dt.to_period('M')
        
        # Get job categories for filtering
        job_categories = ['All Categories'] + sorted(df['Job_Category'].unique().tolist())
        selected_category = st.selectbox("Select Job Category", job_categories, key="trend_category")
        
        if selected_category != 'All Categories':
            df_filtered = df_date[df_date['Job_Category'] == selected_category]
        else:
            df_filtered = df_date
        
        # Create job posting trend by month
        trend_data = df_filtered.groupby(['Month']).size().reset_index(name='Count')
        trend_data['Month'] = trend_data['Month'].astype(str)
        
        # Sort by month chronologically
        trend_data = trend_data.sort_values('Month')
        
        if not trend_data.empty:
            fig_trend = px.line(
                trend_data, 
                x='Month', 
                y='Count',
                title=f"Job Posting Trends Over Time - {selected_category}",
                markers=True
            )
            
            # Add category breakdown if All Categories selected
            if selected_category == 'All Categories':
                cat_trend_data = df_filtered.groupby(['Month', 'Job_Category']).size().reset_index(name='Count')
                cat_trend_data['Month'] = cat_trend_data['Month'].astype(str)
                cat_trend_data = cat_trend_data.sort_values('Month')
                
                fig_trend = px.line(
                    cat_trend_data, 
                    x='Month', 
                    y='Count', 
                    color='Job_Category',
                    title=f"Job Posting Trends Over Time by Category",
                    markers=True
                )
            
            fig_trend.update_layout(
                xaxis_title="Month", 
                yaxis_title="Number of Jobs Posted",
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info(f"No job posting trend data available for {selected_category}")
    else:
        st.info("No date information available for trend analysis")
    
    # Experience and Education Requirements
    st.markdown('<div class="subheader-style">Experience & Education Requirements</div>', unsafe_allow_html=True)
    exp_cols = st.columns(2)
    
    with exp_cols[0]:
        # Experience range distribution
        if df['Experience_Range'].nunique() > 0:
            exp_range_counts = df['Experience_Range'].value_counts().reset_index()
            exp_range_counts.columns = ['Experience Level', 'Count']
            
            # Sort in logical order
            order = [
                "Entry Level (0-1 years)", 
                "Junior (1-3 years)", 
                "Mid-Level (3-5 years)",
                "Senior (5-8 years)", 
                "Expert (8+ years)",
                "Not Specified"
            ]
            exp_range_counts['Experience Level'] = pd.Categorical(
                exp_range_counts['Experience Level'], 
                categories=order, 
                ordered=True
            )
            exp_range_counts = exp_range_counts.sort_values('Experience Level')
            
            fig_exp_range = px.bar(
                exp_range_counts,
                x='Experience Level',
                y='Count',
                title="Distribution of Experience Levels",
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            fig_exp_range.update_traces(textposition='outside')
            fig_exp_range.update_layout(
                xaxis_title="Experience Level", 
                yaxis_title="Number of Jobs",
                xaxis={'categoryorder': 'array', 'categoryarray': order}
            )
            st.plotly_chart(fig_exp_range, use_container_width=True)
        else:
            st.info("No experience level data available")
    
    with exp_cols[1]:
        # Education requirements distribution
        if df['Education_Required'].nunique() > 0:
            edu_counts = df['Education_Required'].value_counts().reset_index()
            edu_counts.columns = ['Education Level', 'Count']
            
            # Define order for education levels
            edu_order = ["PhD", "Master's", "Bachelor's", "Associate's", "Not Specified"]
            
            # Create categorical variable for proper ordering
            edu_counts['Education Level'] = pd.Categorical(
                edu_counts['Education Level'],
                categories=edu_order,
                ordered=True
            )
            
            # Sort based on the categorical order
            edu_counts = edu_counts.sort_values('Education Level')
            
            fig_edu = px.bar(
                edu_counts,
                x='Education Level',
                y='Count',
                title="Education Requirements Distribution",
                color='Education Level',
                color_discrete_sequence=px.colors.qualitative.Bold,
                text='Count'
            )
            fig_edu.update_traces(textposition='outside')
            fig_edu.update_layout(
                xaxis_title="Education Level", 
                yaxis_title="Number of Jobs",
                xaxis={'categoryorder': 'array', 'categoryarray': edu_order}
            )
            st.plotly_chart(fig_edu, use_container_width=True)
        else:
            st.info("No education requirement data available")
    
    # Location Analysis
    if 'Country' in df.columns and df['Country'].nunique() > 1:
        st.markdown('<div class="subheader-style">Job Locations</div>', unsafe_allow_html=True)
        
        country_counts = df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig_country = px.bar(
            country_counts,
            x='Country',
            y='Count',
            title="Jobs by Country",
            color='Count',
            color_continuous_scale='Blues',
            text='Count'
        )
        fig_country.update_traces(textposition='outside')
        fig_country.update_layout(xaxis_title="Location", yaxis_title="Number of Jobs")
        st.plotly_chart(fig_country, use_container_width=True)
    
    # City analysis if multiple cities
    if 'City' in df.columns and df['City'].nunique() > 1:
        cities = df['City'].value_counts().reset_index()
        cities.columns = ['City', 'Count']
        
        # Show top 10 cities only
        top_cities = cities.head(10)
        
        fig_cities = px.bar(
            top_cities,
            x='City',
            y='Count',
            title="Top 10 Cities with Most Job Postings",
            color='Count',
            color_continuous_scale='Teal',
            text='Count'
        )
        fig_cities.update_traces(textposition='outside')
        fig_cities.update_layout(xaxis_title="City", yaxis_title="Number of Jobs")
        st.plotly_chart(fig_cities, use_container_width=True)

@handle_exceptions
def create_skills_analysis_tab(df):
    """Create content for Skills Analysis tab"""
    st.markdown('<div class="subheader-style">Technical Skills Analysis</div>', unsafe_allow_html=True)
    
    # Filter by job category for skills
    job_categories = ['All Categories'] + sorted(df['Job_Category'].unique().tolist())
    selected_category = st.selectbox("Filter skills by job category", job_categories, key="skills_category")
    
    if selected_category != 'All Categories':
        df_filtered = df[df['Job_Category'] == selected_category]
    else:
        df_filtered = df
    
    skills_col1, skills_col2 = st.columns(2)
    
    # Extract all skills mentioned and count their frequency
    all_skills = [skill for skills_list in df_filtered['Skills'] for skill in skills_list]
    skill_counts = Counter(all_skills)
    
    # Convert to DataFrame for visualization
    skill_df = pd.DataFrame({
        'Skill': list(skill_counts.keys()),
        'Count': list(skill_counts.values())
    }).sort_values('Count', ascending=False).head(20)  # Show top 20 skills
    
    with skills_col1:
        if not skill_df.empty:
            fig_skills = px.bar(
                skill_df,
                y='Skill',
                x='Count',
                title=f"Top Skills in Demand ({selected_category})",
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            
            fig_skills.update_traces(textposition='outside')
            fig_skills.update_layout(yaxis_title="Skill", xaxis_title="Number of Job Listings")
            st.plotly_chart(fig_skills, use_container_width=True)
            
            # Add skill search functionality
            st.markdown('<div class="subheader-style">Search for Specific Skills</div>', unsafe_allow_html=True)
            search_skill = st.text_input("Enter a skill to see demand", key="skill_search")
            
            if search_skill:
                matches = [skill for skill in all_skills if search_skill.lower() in skill.lower()]
                if matches:
                    st.success(f"Found {len(matches)} job listings requiring '{search_skill}'")
                else:
                    st.warning(f"No job listings found requiring '{search_skill}'")
        else:
            st.info("No skills data available for the selected category")
    
    with skills_col2:
        # Generate wordcloud for skills
        if all_skills:
            st.markdown("### Skills Word Cloud")
            # Create a string of all skills for the wordcloud
            skill_text = ' '.join(all_skills)
            
            # Generate wordcloud
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white', 
                    colormap='viridis',
                    max_words=100,
                    contour_width=1,
                    contour_color='steelblue'
                ).generate(skill_text)
                
                # Display wordcloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate wordcloud: {str(e)}")
        else:
            st.info("No skills data available for the selected category")
    
    # Skill correlation analysis
    st.markdown('<div class="subheader-style">Soft Skills Analysis</div>', unsafe_allow_html=True)
    
    # Extract all soft skills mentioned and count their frequency
    all_soft_skills = [skill for skills_list in df_filtered['Soft_Skills'] for skill in skills_list]
    soft_skill_counts = Counter(all_soft_skills)
    
    # Convert to DataFrame for visualization
    soft_skill_df = pd.DataFrame({
        'Skill': list(soft_skill_counts.keys()),
        'Count': list(soft_skill_counts.values())
    }).sort_values('Count', ascending=False)
    
    if not soft_skill_df.empty:
        fig_soft_skills = px.bar(
            soft_skill_df,
            y='Skill',
            x='Count',
            title="Soft Skills in Demand",
            color='Count',
            color_continuous_scale='Oranges',
            text='Count'
        )
        fig_soft_skills.update_traces(textposition='outside')
        fig_soft_skills.update_layout(yaxis_title="Soft Skill", xaxis_title="Number of Job Listings")
        st.plotly_chart(fig_soft_skills, use_container_width=True)
    else:
        st.info("No soft skills data available for the selected category")
    
    # Skills by experience level
    st.markdown('<div class="subheader-style">Skills Demand by Experience Level</div>', unsafe_allow_html=True)
    
    # Filter to rows with specified experience
    df_with_exp = df_filtered.dropna(subset=['Experience_Range'])
    
    if not df_with_exp.empty:
        # Create a selector for experience level
        exp_levels = sorted(df_with_exp['Experience_Range'].unique().tolist())
        selected_exp = st.selectbox("Select Experience Level", exp_levels, key="exp_level_skills")
        
        # Filter by selected experience level
        df_exp_filtered = df_with_exp[df_with_exp['Experience_Range'] == selected_exp]
        
        # Extract skills for this experience level
        exp_skills = [skill for skills_list in df_exp_filtered['Skills'] for skill in skills_list]
        exp_skill_counts = Counter(exp_skills)
        
        # Convert to DataFrame for visualization
        exp_skill_df = pd.DataFrame({
            'Skill': list(exp_skill_counts.keys()),
            'Count': list(exp_skill_counts.values())
        }).sort_values('Count', ascending=False).head(15)
        
        if not exp_skill_df.empty:
            fig_exp_skills = px.bar(
                exp_skill_df,
                y='Skill',
                x='Count',
                title=f"Skills in Demand for {selected_exp}",
                color='Count',
                color_continuous_scale='Teal',
                text='Count'
            )
            fig_exp_skills.update_traces(textposition='outside')
            fig_exp_skills.update_layout(yaxis_title="Skill", xaxis_title="Number of Job Listings")
            st.plotly_chart(fig_exp_skills, use_container_width=True)
        else:
            st.info(f"No skills data available for {selected_exp}")
    else:
        st.info("No experience level data available")
    
    # Skills comparison by education level
    st.markdown('<div class="subheader-style">Skills by Education Level</div>', unsafe_allow_html=True)
    
    # Filter to rows with specified education
    df_with_edu = df_filtered.dropna(subset=['Education_Required'])
    df_with_edu = df_with_edu[df_with_edu['Education_Required'] != 'Not Specified']
    
    if not df_with_edu.empty and df_with_edu['Education_Required'].nunique() > 1:
        edu_levels = sorted(df_with_edu['Education_Required'].unique().tolist())
        
        # Create multi-select for education levels
        selected_edu_levels = st.multiselect(
            "Select Education Levels to Compare", 
            edu_levels,
            default=edu_levels[:min(2, len(edu_levels))],
            key="edu_level_skills"
        )
        
        if selected_edu_levels:
            # Create a figure for comparison
            fig_edu_skills = go.Figure()
            
            for edu_level in selected_edu_levels:
                # Filter by education level
                df_edu_filtered = df_with_edu[df_with_edu['Education_Required'] == edu_level]
                
                # Extract skills for this education level
                edu_skills = [skill for skills_list in df_edu_filtered['Skills'] for skill in skills_list]
                edu_skill_counts = Counter(edu_skills)
                
                # Get top 10 skills
                top_skills = dict(edu_skill_counts.most_common(10))
                
                # Add bar chart for this education level
                fig_edu_skills.add_trace(go.Bar(
                    x=list(top_skills.keys()),
                    y=list(top_skills.values()),
                    name=edu_level
                ))
            
            fig_edu_skills.update_layout(
                title=f"Top Skills by Education Level",
                xaxis_title="Skill",
                yaxis_title="Number of Job Listings",
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_edu_skills, use_container_width=True)
        else:
            st.info("Please select at least one education level to view skills")
    else:
        st.info("Not enough education level data to compare skills")

@handle_exceptions
def create_job_categories_tab(df):
    """Create content for Job Categories tab with deep dive analysis"""
    st.markdown('<div class="subheader-style">Job Categories Deep Dive</div>', unsafe_allow_html=True)
    
    # Select job category to analyze
    job_categories = sorted(df['Job_Category'].unique().tolist())
    if job_categories:
        selected_category = st.selectbox("Select Job Category for Deep Analysis", job_categories, key="deep_dive_category")
        
        # Filter data for selected category
        df_category = df[df['Job_Category'] == selected_category]
        
        # Show subcategories if available
        subcategories = df_category['Job_Subcategory'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subcategory Analysis
            if not subcategories.empty and len(subcategories) > 1:
                st.markdown('<div class="subheader-style">Subcategories</div>', unsafe_allow_html=True)
                
                fig_subcat = px.pie(
                    names=subcategories.index,
                    values=subcategories.values,
                    title=f"Subcategories within {selected_category}",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_subcat.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_subcat, use_container_width=True)
            
            # Work mode distribution for this category
            work_modes = df_category['Work_Mode'].value_counts()
            
            st.markdown('<div class="subheader-style">Work Mode</div>', unsafe_allow_html=True)
            fig_work = px.pie(
                names=work_modes.index,
                values=work_modes.values,
                title=f"Work Mode Distribution for {selected_category}",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_work.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_work, use_container_width=True)
        
        with col2:
            # Experience requirements for this category
            st.markdown('<div class="subheader-style">Experience Requirements</div>', unsafe_allow_html=True)
            exp_ranges = df_category['Experience_Range'].value_counts().reset_index()
            exp_ranges.columns = ['Experience Range', 'Count']
            
            # Sort in logical order
            order = [
                "Entry Level (0-1 years)", 
                "Junior (1-3 years)", 
                "Mid-Level (3-5 years)",
                "Senior (5-8 years)", 
                "Expert (8+ years)",
                "Not Specified"
            ]
            exp_ranges['Experience Range'] = pd.Categorical(
                exp_ranges['Experience Range'], 
                categories=order, 
                ordered=True
            )
            exp_ranges = exp_ranges.sort_values('Experience Range')
            
            fig_exp = px.bar(
                exp_ranges,
                x='Experience Range',
                y='Count',
                title=f"Experience Requirements for {selected_category}",
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            fig_exp.update_traces(textposition='outside')
            fig_exp.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': order}
            )
            st.plotly_chart(fig_exp, use_container_width=True)
            
            # Education requirements for this category
            st.markdown('<div class="subheader-style">Education Requirements</div>', unsafe_allow_html=True)
            edu_req = df_category['Education_Required'].value_counts().reset_index()
            edu_req.columns = ['Education', 'Count']
            
            # Define order for education levels
            edu_order = ["PhD", "Master's", "Bachelor's", "Associate's", "Not Specified"]
            
            # Create categorical variable for proper ordering
            edu_req['Education'] = pd.Categorical(
                edu_req['Education'],
                categories=edu_order,
                ordered=True
            )
            
            # Sort based on the categorical order
            edu_req = edu_req.sort_values('Education')
            
            fig_edu = px.bar(
                edu_req,
                x='Education',
                y='Count',
                title=f"Education Requirements for {selected_category}",
                color='Count',
                color_continuous_scale='Blues',
                text='Count'
            )
            fig_edu.update_traces(textposition='outside')
            fig_edu.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': edu_order}
            )
            st.plotly_chart(fig_edu, use_container_width=True)
        
        # Skills specific to this category
        st.markdown('<div class="subheader-style">Skills for this Category</div>', unsafe_allow_html=True)
        
        # Extract skills for this category
        cat_skills = [skill for skills_list in df_category['Skills'] for skill in skills_list]
        cat_skill_counts = Counter(cat_skills)
        
        # Convert to DataFrame for visualization
        cat_skill_df = pd.DataFrame({
            'Skill': list(cat_skill_counts.keys()),
            'Count': list(cat_skill_counts.values())
        }).sort_values('Count', ascending=False).head(20)
        
        if not cat_skill_df.empty:
            fig_skills = px.bar(
                cat_skill_df,
                y='Skill',
                x='Count',
                title=f"Top Skills Required for {selected_category}",
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            fig_skills.update_traces(textposition='outside')
            fig_skills.update_layout(yaxis_title="Skill", xaxis_title="Number of Job Listings")
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.info(f"No skills data available for {selected_category}")
        
        # Subcategory specific analysis
        if 'Job_Subcategory' in df_category.columns and df_category['Job_Subcategory'].nunique() > 1:
            st.markdown('<div class="subheader-style">Subcategory Deep Dive</div>', unsafe_allow_html=True)
            
            # Select subcategory for deeper analysis
            subcats = ['All Subcategories'] + sorted(df_category['Job_Subcategory'].unique().tolist())
            selected_subcat = st.selectbox("Select Subcategory", subcats, key="subcat_select")
            
            if selected_subcat != 'All Subcategories':
                df_subcat = df_category[df_category['Job_Subcategory'] == selected_subcat]
            else:
                df_subcat = df_category
            
            subcat_col1, subcat_col2 = st.columns(2)
            
            with subcat_col1:
                # Experience distribution for subcategory
                subcat_exp = df_subcat['Experience_Range'].value_counts().reset_index()
                subcat_exp.columns = ['Experience Range', 'Count']
                
                # Sort in logical order
                order = [
                    "Entry Level (0-1 years)", 
                    "Junior (1-3 years)", 
                    "Mid-Level (3-5 years)",
                    "Senior (5-8 years)", 
                    "Expert (8+ years)",
                    "Not Specified"
                ]
                
                subcat_exp['Experience Range'] = pd.Categorical(
                    subcat_exp['Experience Range'], 
                    categories=order, 
                    ordered=True
                )
                subcat_exp = subcat_exp.sort_values('Experience Range')
                
                fig_subcat_exp = px.bar(
                    subcat_exp,
                    x='Experience Range',
                    y='Count',
                    title=f"Experience Requirements for {selected_subcat}",
                    color='Count',
                    color_continuous_scale='Purples',
                    text='Count'
                )
                fig_subcat_exp.update_traces(textposition='outside')
                fig_subcat_exp.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': order}
                )
                st.plotly_chart(fig_subcat_exp, use_container_width=True)
            
            with subcat_col2:
                # Extract skills for this subcategory
                subcat_skills = [skill for skills_list in df_subcat['Skills'] for skill in skills_list]
                subcat_skill_counts = Counter(subcat_skills)
                
                # Convert to DataFrame for visualization
                subcat_skill_df = pd.DataFrame({
                    'Skill': list(subcat_skill_counts.keys()),
                    'Count': list(subcat_skill_counts.values())
                }).sort_values('Count', ascending=False).head(10)
                
                if not subcat_skill_df.empty:
                    fig_subcat_skills = px.bar(
                        subcat_skill_df,
                        y='Skill',
                        x='Count',
                        title=f"Top Skills for {selected_subcat}",
                        color='Count',
                        color_continuous_scale='Greens',
                        text='Count'
                    )
                    fig_subcat_skills.update_traces(textposition='outside')
                    fig_subcat_skills.update_layout(yaxis_title="Skill", xaxis_title="Frequency")
                    st.plotly_chart(fig_subcat_skills, use_container_width=True)
                else:
                    st.info(f"No skills data available for {selected_subcat}")
            
            # Show specific job listings for this subcategory
            st.markdown('<div class="subheader-style">Sample Job Listings</div>', unsafe_allow_html=True)
            if not df_subcat.empty:
                # Sample a few jobs to display
                sample_size = min(5, len(df_subcat))
                sample_jobs = df_subcat.sample(sample_size) if sample_size > 0 else df_subcat
                
                for _, job in sample_jobs.iterrows():
                    with st.expander(f"{job['Job Title']} - {job['Location']}"):
                        st.markdown(f"**Company:** {job['Company']}")
                        st.markdown(f"**Category:** {job['Job_Category']} - {job['Job_Subcategory']}")
                        st.markdown(f"**Experience Required:** {job['Experience_Range']}")
                        st.markdown(f"**Education Required:** {job['Education_Required']}")
                        st.markdown(f"**Work Mode:** {job['Work_Mode']}")
                        
                        # Display skills if available
                        if 'Skills' in job and job['Skills']:
                            st.markdown("**Skills Required:**")
                            st.write(", ".join(job['Skills']))
    else:
        st.info("No job categories available for analysis")

@handle_exceptions
def create_company_comparison_tab(df):
    """Create content for company comparison tab"""
    st.markdown('<div class="subheader-style">Company Comparison</div>', unsafe_allow_html=True)
    
    # Check if multiple companies exist in the dataset
    if 'Company' in df.columns and df['Company'].nunique() > 1:
        # Select companies to compare
        all_companies = sorted(df['Company'].unique().tolist())
        selected_companies = st.multiselect(
            "Select Companies to Compare", 
            all_companies,
            default=all_companies[:min(3, len(all_companies))]
        )
        
        if selected_companies:
            # Filter data for selected companies
            df_compare = df[df['Company'].isin(selected_companies)]
            
            # Categories comparison
            st.markdown('<div class="subheader-style">Job Categories by Company</div>', unsafe_allow_html=True)
            
            # Create a dataframe for category comparison
            cat_counts = df_compare.groupby(['Company', 'Job_Category']).size().reset_index(name='Count')
            
            # Create grouped bar chart
            fig_cats = px.bar(
                cat_counts,
                x='Job_Category',
                y='Count',
                color='Company',
                title="Job Categories Comparison",
                barmode='group',
                text='Count'
            )
            fig_cats.update_traces(textposition='outside')
            fig_cats.update_layout(
                xaxis_title="Job Category", 
                yaxis_title="Number of Jobs",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_cats, use_container_width=True)
            
            # Experience requirements comparison
            st.markdown('<div class="subheader-style">Experience Requirements by Company</div>', unsafe_allow_html=True)
            
            # Create a dataframe for experience comparison
            exp_counts = df_compare.groupby(['Company', 'Experience_Range']).size().reset_index(name='Count')
            
            # Sort by experience level
            order = [
                "Entry Level (0-1 years)", 
                "Junior (1-3 years)", 
                "Mid-Level (3-5 years)",
                "Senior (5-8 years)", 
                "Expert (8+ years)",
                "Not Specified"
            ]
            exp_counts['Experience_Range'] = pd.Categorical(
                exp_counts['Experience_Range'], 
                categories=order, 
                ordered=True
            )
            exp_counts = exp_counts.sort_values('Experience_Range')
            
            # Create grouped bar chart
            fig_exp = px.bar(
                exp_counts,
                x='Experience_Range',
                y='Count',
                color='Company',
                title="Experience Requirements Comparison",
                barmode='group',
                text='Count'
            )
            fig_exp.update_traces(textposition='outside')
            fig_exp.update_layout(
                xaxis_title="Experience Level", 
                yaxis_title="Number of Jobs",
                xaxis={'categoryorder': 'array', 'categoryarray': order},
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_exp, use_container_width=True)
            
            # Work mode comparison
            st.markdown('<div class="subheader-style">Work Mode by Company</div>', unsafe_allow_html=True)
            
            # Create a dataframe for work mode comparison
            work_counts = df_compare.groupby(['Company', 'Work_Mode']).size().reset_index(name='Count')
            
            # Create grouped bar chart
            fig_work = px.bar(
                work_counts,
                x='Work_Mode',
                y='Count',
                color='Company',
                title="Remote vs. Onsite Comparison",
                barmode='group',
                text='Count'
            )
            fig_work.update_traces(textposition='outside')
            fig_work.update_layout(
                xaxis_title="Work Mode", 
                yaxis_title="Number of Jobs",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_work, use_container_width=True)
            
            # Education requirements comparison
            st.markdown('<div class="subheader-style">Education Requirements by Company</div>', unsafe_allow_html=True)
            
            # Create a dataframe for education comparison
            edu_counts = df_compare.groupby(['Company', 'Education_Required']).size().reset_index(name='Count')
            
            # Create grouped bar chart
            fig_edu = px.bar(
                edu_counts,
                x='Education_Required',
                y='Count',
                color='Company',
                title="Education Requirements Comparison",
                barmode='group',
                text='Count'
            )
            fig_edu.update_traces(textposition='outside')
            fig_edu.update_layout(
                xaxis_title="Education Required", 
                yaxis_title="Number of Jobs",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_edu, use_container_width=True)
            
            # Skills comparison
            st.markdown('<div class="subheader-style">Top 5 Skills by Company</div>', unsafe_allow_html=True)
            
            # Get top 5 skills for each company
            company_skills = {}
            for company in selected_companies:
                df_company = df[df['Company'] == company]
                skills = [skill for skills_list in df_company['Skills'] for skill in skills_list]
                top_skills = Counter(skills).most_common(5)
                company_skills[company] = top_skills
            
            # Create a table for comparison
            skills_data = []
            for company, top_skills in company_skills.items():
                for i, (skill, count) in enumerate(top_skills):
                    skills_data.append({
                        'Company': company,
                        'Rank': i + 1,
                        'Skill': skill,
                        'Count': count
                    })
            
            skills_df = pd.DataFrame(skills_data)
            
            # Pivot for better visualization
            skills_pivot = skills_df.pivot(index='Rank', columns='Company', values='Skill')
            st.dataframe(skills_pivot, use_container_width=True)
            
            # Heatmap of skills across companies
            st.markdown('<div class="subheader-style">Skills Distribution Heatmap</div>', unsafe_allow_html=True)
            
            # Get common skills across companies
            all_skills = set()
            for company in selected_companies:
                df_company = df[df['Company'] == company]
                company_skills = set([skill for skills_list in df_company['Skills'] for skill in skills_list])
                all_skills.update(company_skills)
            
            # Create matrix of skill counts by company
            skill_matrix = []
            for skill in all_skills:
                row = {'Skill': skill}
                for company in selected_companies:
                    df_company = df[df['Company'] == company]
                    count = sum(1 for skills_list in df_company['Skills'] if skill in skills_list)
                    row[company] = count
                skill_matrix.append(row)
            
            skill_matrix_df = pd.DataFrame(skill_matrix)
            
            # Sort by most common skills
            skill_matrix_df['Total'] = skill_matrix_df[selected_companies].sum(axis=1)
            skill_matrix_df = skill_matrix_df.sort_values('Total', ascending=False).head(20)  # Top 20 skills
            skill_matrix_df = skill_matrix_df.drop(columns=['Total'])
            
            # Melt for heatmap format
            skill_heatmap_df = skill_matrix_df.melt(
                id_vars=['Skill'], 
                value_vars=selected_companies,
                var_name='Company', 
                value_name='Count'
            )
            
            # Create heatmap
            fig_heatmap = px.density_heatmap(
                skill_heatmap_df,
                x='Company',
                y='Skill',
                z='Count',
                title="Skill Distribution Across Companies",
                color_continuous_scale='Viridis'
            )
            fig_heatmap.update_layout(
                xaxis_title="Company",
                yaxis_title="Skill",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Please select at least one company to compare")
    else:
        st.info("Multiple company comparison requires data from more than one company")

@handle_exceptions
def create_compensation_tab(df):
    """Create content for compensation and benefits tab"""
    st.markdown('<div class="subheader-style">Compensation & Benefits Analysis</div>', unsafe_allow_html=True)
    
    # Check if any salary data is available
    has_salary_data = (
        'Min_Salary' in df.columns and 
        'Max_Salary' in df.columns and 
        df['Min_Salary'].notna().any() and 
        df['Max_Salary'].notna().any()
    )
    
    if has_salary_data:
        # Salary range distribution
        salary_df = df.dropna(subset=['Min_Salary', 'Max_Salary'])
        
        if not salary_df.empty:
            st.markdown('<div class="subheader-style">Salary Ranges</div>', unsafe_allow_html=True)
            
            # Create salary bins
            salary_bins = [0, 50000, 75000, 100000, 125000, 150000, 175000, 200000, float('inf')]
            bin_labels = ['<$50K', '$50K-$75K', '$75K-$100K', '$100K-$125K', '$125K-$150K', '$150K-$175K', '$175K-$200K', '$200K+']
            
            # Bin the average salaries
            salary_df['Salary_Bin'] = pd.cut(
                salary_df['Avg_Salary'], 
                bins=salary_bins, 
                labels=bin_labels, 
                right=False
            )
            
            # Count jobs in each bin
            salary_counts = salary_df['Salary_Bin'].value_counts().reset_index()
            salary_counts.columns = ['Salary Range', 'Count']
            
            # Sort by salary range
            salary_counts['Salary Range'] = pd.Categorical(
                salary_counts['Salary Range'], 
                categories=bin_labels, 
                ordered=True
            )
            salary_counts = salary_counts.sort_values('Salary Range')
            
            fig_salary = px.bar(
                salary_counts,
                x='Salary Range',
                y='Count',
                title="Salary Range Distribution",
                color='Count',
                color_continuous_scale='Greens',
                text='Count'
            )
            fig_salary.update_traces(textposition='outside')
            fig_salary.update_layout(
                xaxis_title="Salary Range", 
                yaxis_title="Number of Jobs",
                xaxis={'categoryorder': 'array', 'categoryarray': bin_labels}
            )
            st.plotly_chart(fig_salary, use_container_width=True)
            
            # Salary by job category
            st.markdown('<div class="subheader-style">Salary by Job Category</div>', unsafe_allow_html=True)
            
            # Aggregate salaries by job category
            cat_salary = salary_df.groupby('Job_Category').agg({
                'Min_Salary': 'mean',
                'Max_Salary': 'mean',
                'Avg_Salary': 'mean'
            }).reset_index()
            
            # Format salaries
            cat_salary['Min_Salary'] = cat_salary['Min_Salary'].round(0)
            cat_salary['Max_Salary'] = cat_salary['Max_Salary'].round(0)
            cat_salary['Avg_Salary'] = cat_salary['Avg_Salary'].round(0)
            
            # Sort by average salary
            cat_salary = cat_salary.sort_values('Avg_Salary', ascending=False)
            
            # Create bar chart
            fig_cat_salary = px.bar(
                cat_salary,
                x='Job_Category',
                y='Avg_Salary',
                title="Average Salary by Job Category",
                color='Avg_Salary',
                color_continuous_scale='Plasma',
                text=cat_salary['Avg_Salary'].apply(lambda x: f"${x:,.0f}")
            )
            fig_cat_salary.update_traces(textposition='outside')
            fig_cat_salary.update_layout(
                xaxis_title="Job Category", 
                yaxis_title="Average Salary ($)",
                yaxis_tickformat="$,.0f"
            )
            st.plotly_chart(fig_cat_salary, use_container_width=True)
            
            # Salary by experience level
            st.markdown('<div class="subheader-style">Salary by Experience Level</div>', unsafe_allow_html=True)
            
            # Aggregate salaries by experience range
            exp_salary = salary_df.groupby('Experience_Range').agg({
                'Min_Salary': 'mean',
                'Max_Salary': 'mean',
                'Avg_Salary': 'mean'
            }).reset_index()
            
            # Format salaries
            exp_salary['Min_Salary'] = exp_salary['Min_Salary'].round(0)
            exp_salary['Max_Salary'] = exp_salary['Max_Salary'].round(0)
            exp_salary['Avg_Salary'] = exp_salary['Avg_Salary'].round(0)
            
            # Sort by experience level
            order = [
                "Entry Level (0-1 years)", 
                "Junior (1-3 years)", 
                "Mid-Level (3-5 years)",
                "Senior (5-8 years)", 
                "Expert (8+ years)",
                "Not Specified"
            ]
            exp_salary['Experience_Range'] = pd.Categorical(
                exp_salary['Experience_Range'], 
                categories=order, 
                ordered=True
            )
            exp_salary = exp_salary.sort_values('Experience_Range')
            
            # Create box plot to show salary ranges
            fig_exp_salary = go.Figure()
            
            for i, row in exp_salary.iterrows():
                fig_exp_salary.add_trace(go.Box(
                    y=[row['Min_Salary'], row['Avg_Salary'], row['Max_Salary']],
                    name=row['Experience_Range'],
                    boxpoints=False,
                    marker_color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
                ))
            
            fig_exp_salary.update_layout(
                title="Salary Ranges by Experience Level",
                xaxis_title="Experience Level",
                yaxis_title="Salary ($)",
                yaxis_tickformat="$,.0f",
                boxmode='group',
                showlegend=False
            )
            st.plotly_chart(fig_exp_salary, use_container_width=True)
        else:
            st.info("No salary data available for analysis")
    else:
        st.info("No salary information found in the dataset")
    
    # Benefits analysis
    st.markdown('<div class="subheader-style">Benefits Analysis</div>', unsafe_allow_html=True)
    
    if 'Benefits' in df.columns and any(len(benefits) > 0 for benefits in df['Benefits']):
        # Extract all benefits mentioned
        all_benefits = [benefit for benefits_list in df['Benefits'] for benefit in benefits_list]
        benefit_counts = Counter(all_benefits)
        
        # Convert to DataFrame
        benefit_df = pd.DataFrame({
            'Benefit': list(benefit_counts.keys()),
            'Count': list(benefit_counts.values())
        }).sort_values('Count', ascending=False)
        
        if not benefit_df.empty:
            # Create horizontal bar chart
            fig_benefits = px.bar(
                benefit_df,
                y='Benefit',
                x='Count',
                title="Top Benefits Mentioned in Job Postings",
                color='Count',
                color_continuous_scale='Teal',
                text='Count'
            )
            fig_benefits.update_traces(textposition='outside')
            fig_benefits.update_layout(yaxis_title="Benefit", xaxis_title="Number of Job Listings")
            st.plotly_chart(fig_benefits, use_container_width=True)
            
            # Benefits by job category
            st.markdown('<div class="subheader-style">Benefits by Job Category</div>', unsafe_allow_html=True)
            
            # Get top job categories
            top_categories = df['Job_Category'].value_counts().head(5).index.tolist()
            
            # Create a matrix of benefit counts by category
            benefit_matrix = []
            for benefit in benefit_df['Benefit'].head(10):  # Top 10 benefits
                row = {'Benefit': benefit}
                for category in top_categories:
                    df_category = df[df['Job_Category'] == category]
                    count = sum(1 for benefits_list in df_category['Benefits'] if benefit in benefits_list)
                    row[category] = count
                benefit_matrix.append(row)
            
            benefit_matrix_df = pd.DataFrame(benefit_matrix)
            
            # Melt for heatmap format
            benefit_heatmap_df = benefit_matrix_df.melt(
                id_vars=['Benefit'], 
                value_vars=top_categories,
                var_name='Job Category', 
                value_name='Count'
            )
            
            # Create heatmap
            fig_benefit_heatmap = px.density_heatmap(
                benefit_heatmap_df,
                x='Job Category',
                y='Benefit',
                z='Count',
                title="Benefit Distribution Across Job Categories",
                color_continuous_scale='Blues'
            )
            fig_benefit_heatmap.update_layout(
                xaxis_title="Job Category",
                yaxis_title="Benefit",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_benefit_heatmap, use_container_width=True)
        else:
            st.info("No benefit data available for analysis")
    else:
        st.info("No benefits information found in the dataset")

@handle_exceptions
def create_job_details_tab(df):
    """Create content for job details tab"""
    st.markdown('<div class="subheader-style">Job Details</div>', unsafe_allow_html=True)
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Job Category filter
        job_categories = ['All Categories'] + sorted(df['Job_Category'].unique().tolist())
        selected_category = st.selectbox("Filter by Job Category", job_categories, key="details_category")
    
    with filter_col2:
        # Experience Range filter
        exp_ranges = ['All Experience Levels'] + sorted(df['Experience_Range'].unique().tolist())
        selected_exp = st.selectbox("Filter by Experience Required", exp_ranges, key="details_exp")
    
    with filter_col3:
        # Work Mode filter
        work_modes = ['All Work Modes'] + sorted(df['Work_Mode'].unique().tolist())
        selected_work = st.selectbox("Filter by Work Mode", work_modes, key="details_work")
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_category != 'All Categories':
        df_filtered = df_filtered[df_filtered['Job_Category'] == selected_category]
    
    if selected_exp != 'All Experience Levels':
        df_filtered = df_filtered[df_filtered['Experience_Range'] == selected_exp]
    
    if selected_work != 'All Work Modes':
        df_filtered = df_filtered[df_filtered['Work_Mode'] == selected_work]
    
    # Add free text search
    search_query = st.text_input("Search in job titles and descriptions", key="job_search")
    if search_query:
        query_lower = search_query.lower()
        df_filtered = df_filtered[
            df_filtered['Job Title'].str.lower().str.contains(query_lower, na=False) | 
            df_filtered['Description'].str.lower().str.contains(query_lower, na=False)
        ]
    
    # Display filterable job listing table
    if not df_filtered.empty:
        # Select columns to display
        display_columns = [
            'Company', 'Job Title', 'Location', 'Job_Category', 'Job_Subcategory',
            'Experience_Range', 'Education_Required', 'Work_Mode', 'Date Posted'
        ]
        
        # Ensure columns exist
        display_columns = [col for col in display_columns if col in df_filtered.columns]
        
        # Format date for display
        if 'Date Posted' in df_filtered.columns:
            df_filtered['Date Posted'] = df_filtered['Date Posted'].dt.strftime('%Y-%m-%d')
        
        # Display table
        st.markdown(f"### Showing {len(df_filtered)} job listings")
        st.dataframe(df_filtered[display_columns], use_container_width=True)
        
        # Show selected job details
        st.markdown('<div class="subheader-style">Job Detail View</div>', unsafe_allow_html=True)
        
        # Get job titles for selection
        job_titles = df_filtered['Job Title'].tolist()
        
        if job_titles:
            selected_job = st.selectbox("Select Job to View Details", job_titles, key="job_detail_select")
            
            # Get the selected job
            job_data = df_filtered[df_filtered['Job Title'] == selected_job].iloc[0]
            
            # Display job details in a card-like format
            st.markdown("""
            <style>
            .job-card {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                background-color: #f9f9f9;
            }
            .job-header {
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .job-section {
                margin-bottom: 15px;
            }
            .job-section-title {
                font-weight: bold;
                color: #555;
                margin-bottom: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="job-card">
                <div class="job-header">
                    <h2>{job_data['Job Title']}</h2>
                    <p><strong>{job_data['Company']}</strong> • {job_data['Location']}</p>
                </div>
                
                <div class="job-section">
                    <div class="job-section-title">Job Details</div>
                    <p><strong>Category:</strong> {job_data['Job_Category']} - {job_data['Job_Subcategory']}</p>
                    <p><strong>Work Mode:</strong> {job_data['Work_Mode']}</p>
                    <p><strong>Experience Required:</strong> {job_data['Experience_Range']}</p>
                    <p><strong>Education Required:</strong> {job_data['Education_Required']}</p>
                    <p><strong>Posted:</strong> {job_data['Date Posted']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display tabs for different sections of the job posting
            job_tabs = st.tabs(["Description", "Responsibilities", "Qualifications", "Skills & Benefits"])
            
            with job_tabs[0]:
                # Display description
                if 'Description' in job_data and isinstance(job_data['Description'], str) and job_data['Description'].strip():
                    st.markdown(job_data['Description'])
                else:
                    st.info("No description available")
            
            with job_tabs[1]:
                # Display responsibilities
                if 'Responsibilities' in job_data and isinstance(job_data['Responsibilities'], str) and job_data['Responsibilities'].strip():
                    st.markdown(job_data['Responsibilities'])
                else:
                    st.info("No responsibilities information available")
            
            with job_tabs[2]:
                # Display qualifications
                min_qual = ""
                if 'Minimum Qualifications' in job_data and isinstance(job_data['Minimum Qualifications'], str) and job_data['Minimum Qualifications'].strip():
                    min_qual = job_data['Minimum Qualifications']
                
                pref_qual = ""
                if 'Preferred Qualifications' in job_data and isinstance(job_data['Preferred Qualifications'], str) and job_data['Preferred Qualifications'].strip():
                    pref_qual = job_data['Preferred Qualifications']
                
                if min_qual or pref_qual:
                    if min_qual:
                        st.markdown("#### Minimum Qualifications")
                        st.markdown(min_qual)
                    
                    if pref_qual:
                        st.markdown("#### Preferred Qualifications")
                        st.markdown(pref_qual)
                else:
                    st.info("No qualifications information available")
            
            with job_tabs[3]:
                # Display skills and benefits
                skills_col, benefits_col = st.columns(2)
                
                with skills_col:
                    st.markdown("#### Technical Skills")
                    if 'Skills' in job_data and job_data['Skills']:
                        for skill in job_data['Skills']:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No technical skills information available")
                    
                    st.markdown("#### Soft Skills")
                    if 'Soft_Skills' in job_data and job_data['Soft_Skills']:
                        for skill in job_data['Soft_Skills']:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No soft skills information available")
                
                with benefits_col:
                    st.markdown("#### Benefits")
                    if 'Benefits' in job_data and job_data['Benefits']:
                        for benefit in job_data['Benefits']:
                            st.markdown(f"- {benefit}")
                    else:
                        st.info("No benefits information available")
                    
                    # Show salary if available
                    st.markdown("#### Compensation")
                    if 'Min_Salary' in job_data and 'Max_Salary' in job_data and pd.notna(job_data['Min_Salary']) and pd.notna(job_data['Max_Salary']):
                        st.markdown(f"Salary Range: ${job_data['Min_Salary']:,.0f} - ${job_data['Max_Salary']:,.0f}")
                    else:
                        st.info("No salary information available")
        else:
            st.info("No jobs match the selected filters")
    else:
        st.info("No jobs match the selected filters")

    # Add export functionality
    if not df_filtered.empty:
        st.markdown("### Export Filtered Results")
        st.markdown(get_table_download_link(df_filtered, "filtered_jobs.csv", "📥 Download Filtered Job Listings"), unsafe_allow_html=True)

# Main function
@handle_exceptions
def main():
    st.sidebar.header("Job Market Analysis")
    st.sidebar.markdown("---")
    
    # Add info about the dashboard
    with st.sidebar.expander("About this Dashboard", expanded=False):
        st.markdown("""
        This dashboard analyzes job posting data to provide insights on:
        
        - Job categories and trends
        - Required skills and qualifications
        - Experience and education requirements
        - Work arrangements (remote, hybrid, onsite)
        - Salary ranges and benefits
        - Company comparisons
        
        Upload your job data CSV files to get started!
        """)
    
    st.sidebar.markdown("---")
    
    # File uploader for multiple CSV files
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV file(s) with job data", 
        type=["csv"], 
        accept_multiple_files=True
    )
    
    st.sidebar.markdown("---")
    
    if uploaded_files:
        # Process all uploaded files
        all_data_frames = []
        
        for uploaded_file in uploaded_files:
            try:
                df = load_and_clean_data(uploaded_file)
                if df is not None:
                    all_data_frames.append(df)
                    st.sidebar.success(f"✅ Successfully loaded data from {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        
        if all_data_frames:
            # Combine all data frames
            combined_df = pd.concat(all_data_frames, ignore_index=True)
            
            # Check if we have valid data
            if not combined_df.empty:
                # Create the dashboard
                create_dashboard(combined_df)
            else:
                st.error("No valid data found in the uploaded files")
        else:
            st.error("Could not load any data from the provided files")
    else:
        # Display instructions
        st.markdown("""
        # 📊 Job Market Analysis Dashboard
        
        This interactive dashboard helps you analyze job market trends across multiple companies.
        
        ## 🚀 How to use:
        1. Upload one or more CSV files containing job data using the sidebar
        2. Each file should contain columns such as: 
           - Job Title
           - Location
           - Date Posted
           - Description
           - Responsibilities
           - Minimum Qualifications
           - Preferred Qualifications
        
        3. The dashboard will automatically analyze and visualize the data
        
        ## ✨ Features:
        - 📊 Analyze job categories and subcategories
        - 🔄 Compare skills requirements across companies
        - 📈 Track experience and education requirements
        - 🧠 Identify in-demand technical and soft skills
        - 💼 View detailed job listings
        - 💰 Analyze compensation and benefits
        """)

def load_and_clean_data_from_df(df):
    """Process a dataframe as if it was loaded from a file"""
    # Fill missing values
    df['Location'] = df['Location'].fillna('Unknown')
    df['Job Title'] = df['Job Title'].fillna('Unknown')
    df['Description'] = df['Description'].fillna('')
    df['Responsibilities'] = df['Responsibilities'].fillna('')
    df['Minimum Qualifications'] = df['Minimum Qualifications'].fillna('')
    df['Preferred Qualifications'] = df['Preferred Qualifications'].fillna('')
    
    # Process location - Extract city and country if available
    df['City'] = df['Location'].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) and "," in x else x
    )
    
    # Extract country from location if available
    df['Country'] = df['Location'].apply(
        lambda x: x.split(",")[-1].strip() if isinstance(x, str) and "," in x else 'Unknown'
    )
    
    # Clean and standardize job titles
    df['Clean_Job_Title'] = df['Job Title'].apply(clean_job_title)
    
    # Extract job categories
    df['Job_Category'] = df['Job Title'].apply(extract_job_category)
    
    # Extract job subcategories for deeper analysis
    df['Job_Subcategory'] = df.apply(
        lambda row: extract_subcategory(row['Job Title'], row['Job_Category']), 
        axis=1
    )
    
    # Convert Date Posted to datetime
    df['Date Posted'] = pd.to_datetime(df['Date Posted'], errors='coerce')
    
    # Set current date for rows with missing date
    current_date = datetime.now()
    df['Date Posted'] = df['Date Posted'].fillna(current_date)
    
    # Create full text field for comprehensive text analysis
    df['Full_Text'] = df.apply(
        lambda row: ' '.join([
            str(row['Job Title']), 
            str(row['Description']), 
            str(row['Responsibilities']), 
            str(row['Minimum Qualifications']), 
            str(row['Preferred Qualifications'])
        ]),
        axis=1
    )
    
    # Extract years of experience required
    df['Experience_Required'] = df.apply(
        lambda row: extract_experience_requirement(str(row['Minimum Qualifications'])), 
        axis=1
    )
    
    # Group experience into ranges for easier analysis
    df['Experience_Range'] = df['Experience_Required'].apply(lambda x: experience_to_range(x))
    
    # Extract education requirements
    df['Education_Required'] = df.apply(
        lambda row: extract_education_requirements(
            str(row['Minimum Qualifications']) + " " + str(row['Preferred Qualifications'])
        ),
        axis=1
    )
    
    # Extract skills from job descriptions and qualifications
    df['Skills'] = df.apply(
        lambda row: extract_skills(row['Full_Text']),
        axis=1
    )
    
    # Extract soft skills and competencies
    soft_skills = [
        'communication', 'teamwork', 'leadership', 'problem solving', 'analytical', 
        'creativity', 'adaptability', 'time management', 'critical thinking', 'collaboration',
        'interpersonal', 'presentation', 'negotiation', 'conflict resolution', 'customer service',
        'emotional intelligence', 'attention to detail', 'organization', 'prioritization',
        'decision making', 'mentoring', 'coaching', 'strategic thinking', 'initiative'
    ]
    
    df['Soft_Skills'] = df.apply(
        lambda row: extract_key_terms(row['Full_Text'], soft_skills),
        axis=1
    )
    
    # Create Remote/Onsite/Hybrid flag
    df['Work_Mode'] = df.apply(
        lambda row: extract_work_mode(row['Full_Text']),
        axis=1
    )
    
    # Extract potential salary information
    df['Min_Salary'], df['Max_Salary'] = zip(*df['Full_Text'].apply(extract_salary_range))
    
    # Calculate average salary when both min and max are available
    df['Avg_Salary'] = df.apply(
        lambda row: (row['Min_Salary'] + row['Max_Salary']) / 2 if pd.notna(row['Min_Salary']) and pd.notna(row['Max_Salary']) else None,
        axis=1
    )
    
    # Extract benefits
    df['Benefits'] = df['Full_Text'].apply(extract_job_benefits)
    
    # Add data quality indicators
    df['Data_Completeness'] = df.apply(
        lambda row: calculate_completeness(row),
        axis=1
    )
    
    return df

if __name__ == "__main__":
    main()