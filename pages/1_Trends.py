import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import final  # Import your existing functions

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Job Market Trends", layout="wide")

# DATA_DIR - consistent with main app
DATA_DIR = "data"

def load_all_company_data(data_dir):
    """Load and process data from all companies"""
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                company_name = file.split('_')[0].capitalize()
                df['Company'] = company_name
                processed_df = final.load_and_clean_data_from_df(df)
                all_data.append(processed_df)
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def create_trending_skills_viz(df):
    """Create visualization for trending skills across all companies"""
    st.markdown("### 🔥 Trending Technical Skills")
    
    # Extract all skills mentioned and count their frequency
    all_skills = [skill for skills_list in df['Skills'] for skill in skills_list]
    skill_counts = Counter(all_skills).most_common(20)
    
    # Convert to DataFrame for visualization
    skill_df = pd.DataFrame(skill_counts, columns=['Skill', 'Count'])
    
    # Create bar chart
    fig = px.bar(
        skill_df,
        y='Skill',
        x='Count',
        title="Top 20 In-Demand Skills Across All Companies",
        color='Count',
        color_continuous_scale='Viridis',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_title="Skill", xaxis_title="Number of Job Listings")
    st.plotly_chart(fig, use_container_width=True)
    
    # Create word cloud
    col1, col2 = st.columns([2, 1])
    with col1:
        skill_text = ' '.join(all_skills)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            colormap='viridis',
            max_words=100
        ).generate(skill_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.write("### Key Insights")
        st.write("Top skills in demand across all analyzed companies:")
        for skill, count in skill_counts[:5]:
            st.write(f"- **{skill}**: {count} listings")

def create_trending_roles_viz(df):
    """Create visualization for trending job roles"""
    st.markdown("### 🚀 Trending Job Roles")
    
    # Get job category and subcategory distribution
    job_cat_counts = df['Job_Category'].value_counts().reset_index()
    job_cat_counts.columns = ['Job Category', 'Count']
    
    # For subcategories, we'll get the top subcategories in each main category
    top_categories = job_cat_counts['Job Category'].head(5).tolist()
    subcategory_data = []
    
    for category in top_categories:
        df_cat = df[df['Job_Category'] == category]
        subcat_counts = df_cat['Job_Subcategory'].value_counts().head(3)
        
        for subcat, count in subcat_counts.items():
            if subcat != 'General':  # Skip generic subcategories
                subcategory_data.append({
                    'Main Category': category,
                    'Subcategory': subcat,
                    'Count': count
                })
    
    subcat_df = pd.DataFrame(subcategory_data)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Main categories chart
        fig1 = px.pie(
            job_cat_counts.head(8),  # Top 8 categories
            values='Count',
            names='Job Category',
            title="Top Job Categories",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Hot subcategories chart
        if not subcat_df.empty:
            fig2 = px.bar(
                subcat_df,
                x='Count',
                y='Subcategory',
                color='Main Category',
                title="Trending Job Specializations",
                orientation='h'
            )
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

def create_trending_locations_viz(df):
    """Create visualization for job locations"""
    st.markdown("### 🌎 Trending Locations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Countries chart
        if 'Country' in df.columns and df['Country'].nunique() > 1:
            country_counts = df['Country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            
            fig1 = px.bar(
                country_counts.head(10),
                x='Country',
                y='Count',
                title="Top Countries for Job Opportunities",
                color='Count',
                color_continuous_scale='Blues',
                text='Count'
            )
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cities chart
        if 'City' in df.columns and df['City'].nunique() > 1:
            city_counts = df['City'].value_counts().reset_index()
            city_counts.columns = ['City', 'Count']
            
            fig2 = px.bar(
                city_counts.head(10),
                x='City',
                y='Count',
                title="Top Cities for Job Opportunities",
                color='Count',
                color_continuous_scale='Teal',
                text='Count'
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

def create_experience_trends_viz(df):
    """Create visualization for experience level trends"""
    st.markdown("### 👔 Experience Level Demand")
    
    # Experience range distribution
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
    
    fig = px.bar(
        exp_range_counts,
        x='Experience Level',
        y='Count',
        title="Experience Level Demand Across All Companies",
        color='Count',
        color_continuous_scale='Viridis',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Experience Level", 
        yaxis_title="Number of Jobs",
        xaxis={'categoryorder': 'array', 'categoryarray': order}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add trends by company
    st.markdown("#### Experience Level Demand by Top Companies")
    top_companies = df['Company'].value_counts().head(5).index.tolist()
    
    # Filter for top companies
    df_top = df[df['Company'].isin(top_companies)]
    
    # Group by company and experience level
    exp_by_company = df_top.groupby(['Company', 'Experience_Range']).size().reset_index(name='Count')
    
    # Create company comparison chart
    fig2 = px.bar(
        exp_by_company,
        x='Experience_Range',
        y='Count',
        color='Company',
        title="Experience Requirements by Company",
        barmode='group'
    )
    fig2.update_layout(
        xaxis_title="Experience Level",
        yaxis_title="Number of Jobs"
    )
    st.plotly_chart(fig2, use_container_width=True)

def create_soft_skills_viz(df):
    """Create visualization for soft skills trends"""
    st.markdown("### 🤝 Trending Soft Skills")
    
    # Extract all soft skills
    all_soft_skills = [skill for skills_list in df['Soft_Skills'] for skill in skills_list]
    soft_skill_counts = Counter(all_soft_skills).most_common(15)
    
    # Convert to DataFrame
    soft_df = pd.DataFrame(soft_skill_counts, columns=['Skill', 'Count'])
    
    fig = px.bar(
        soft_df,
        y='Skill',
        x='Count',
        title="Most Valued Soft Skills Across All Companies",
        color='Count',
        color_continuous_scale='Oranges',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_title="Soft Skill", xaxis_title="Number of Job Listings")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for the trends dashboard"""
    st.markdown('<h1 style="text-align: center;">Cross-Company Job Market Trends</h1>', unsafe_allow_html=True)
    st.write("Analysis of trends and patterns across all companies in the job market")
    
    # Load data from all companies
    df = load_all_company_data(DATA_DIR)
    
    if df is not None:
        # Key metrics
        st.markdown("## 📊 Market Overview")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Jobs Analyzed", len(df))
        
        with metric_cols[1]:
            st.metric("Companies", df['Company'].nunique())
        
        with metric_cols[2]:
            st.metric("Job Categories", df['Job_Category'].nunique())
        
        with metric_cols[3]:
            remote_pct = round(100 * len(df[df['Work_Mode'] == 'Remote']) / len(df), 1)
            st.metric("Remote Jobs", f"{remote_pct}%")
        
        # Create trend visualizations
        create_trending_skills_viz(df)
        create_trending_roles_viz(df)
        create_trending_locations_viz(df)
        create_experience_trends_viz(df)
        create_soft_skills_viz(df)
        
    else:
        st.error("No data available. Please add company data files to the data directory.")

if __name__ == "__main__":
    main()