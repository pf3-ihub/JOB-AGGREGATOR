import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
from collections import Counter
from datetime import datetime, timedelta
import final  # Import your existing functions

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Job Market Predictions", layout="wide")

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

def predict_job_category_growth(df):
    """Predict job category growth based on historical data"""
    if 'Date Posted' not in df.columns or df['Date Posted'].isna().all():
        st.warning("Date information not available for trend prediction")
        return
    
    # Create month data
    df['Month'] = pd.to_datetime(df['Date Posted']).dt.to_period('M')
    
    # Get job category counts by month
    category_by_month = df.groupby(['Month', 'Job_Category']).size().reset_index(name='Count')
    
    # Convert Month to datetime for prediction
    category_by_month['Month_Num'] = category_by_month['Month'].apply(lambda x: x.to_timestamp().timestamp())
    
    # Get top categories
    top_categories = df['Job_Category'].value_counts().head(5).index.tolist()
    
    # Predict growth for each category
    predictions = []
    prediction_months = 6  # Predict next 6 months
    
    for category in top_categories:
        cat_data = category_by_month[category_by_month['Job_Category'] == category]
        
        if len(cat_data) < 3:  # Need at least 3 data points
            continue
        
        # Prepare data for prediction
        X = cat_data['Month_Num'].values.reshape(-1, 1)
        y = cat_data['Count'].values
        
        # Create polynomial features for non-linear trends
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate future months for prediction
        last_month = cat_data['Month'].max()
        future_months = []
        future_timestamps = []
        
        for i in range(1, prediction_months + 1):
            # Add months to the last month
            next_month = pd.Period(last_month) + i
            future_months.append(str(next_month))
            future_timestamps.append(next_month.to_timestamp().timestamp())
        
        # Predict future values
        future_X = np.array(future_timestamps).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        future_y = model.predict(future_X_poly)
        
        # Calculate growth percentage
        current_count = cat_data['Count'].iloc[-1]
        predicted_count = future_y[-1]
        growth_pct = ((predicted_count - current_count) / current_count) * 100 if current_count > 0 else 0
        
        # Store predictions
        predictions.append({
            'Category': category,
            'Current Count': current_count,
            'Predicted Count': round(predicted_count, 1),
            'Growth %': round(growth_pct, 1),
            'Future Months': future_months,
            'Future Values': future_y
        })
    
    # Sort by growth percentage
    predictions.sort(key=lambda x: x['Growth %'], reverse=True)
    
    # Visualize predictions
    st.markdown("### Predicted Job Category Growth (Next 6 Months)")
    
    # Create a bar chart of growth percentages
    growth_df = pd.DataFrame([{
        'Category': p['Category'],
        'Growth %': p['Growth %']
    } for p in predictions])
    
    fig1 = px.bar(
        growth_df,
        x='Category',
        y='Growth %',
        title="Predicted Growth by Job Category",
        color='Growth %',
        color_continuous_scale='RdYlGn',
        text='Growth %'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Create line charts showing predictions
    st.markdown("### Job Category Trend Projections")
    
    for pred in predictions:
        category = pred['Category']
        
        # Get historical data
        cat_hist = category_by_month[category_by_month['Job_Category'] == category]
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=cat_hist['Month'].astype(str),
            y=cat_hist['Count'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add predicted data
        fig.add_trace(go.Scatter(
            x=pred['Future Months'],
            y=pred['Future Values'],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Prediction for {category} (Growth: {pred['Growth %']}%)",
            xaxis_title="Month",
            yaxis_title="Number of Jobs",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def predict_skill_demand(df):
    """Predict emerging skills based on trend analysis"""
    # Create month data if dates are available
    if 'Date Posted' in df.columns and not df['Date Posted'].isna().all():
        df['Month'] = pd.to_datetime(df['Date Posted']).dt.to_period('M')
        
        # Get all unique months
        all_months = sorted(df['Month'].unique())
        
        if len(all_months) >= 3:  # Need at least 3 months of data
            # Analyze skill growth over time
            skill_growth = []
            
            # Get all skills
            all_skills = set()
            for skills_list in df['Skills']:
                all_skills.update(skills_list)
            
            # Filter to skills that appear enough times
            skill_counts = Counter([skill for skills_list in df['Skills'] for skill in skills_list])
            relevant_skills = [skill for skill, count in skill_counts.items() if count >= 5]
            
            # Analyze trend for each skill
            for skill in relevant_skills:
                # Count occurrences by month
                monthly_counts = []
                
                for month in all_months:
                    df_month = df[df['Month'] == month]
                    count = sum(1 for skills_list in df_month['Skills'] if skill in skills_list)
                    monthly_counts.append(count)
                
                # Calculate growth trend
                if len(monthly_counts) >= 2 and sum(monthly_counts) > 0:
                    # Simple growth rate (last month vs first month)
                    first_count = monthly_counts[0] if monthly_counts[0] > 0 else 1
                    last_count = monthly_counts[-1]
                    growth_rate = (last_count - first_count) / first_count
                    
                    # Calculate acceleration (is growth accelerating?)
                    if len(monthly_counts) >= 3:
                        first_half_growth = (monthly_counts[len(monthly_counts)//2] - monthly_counts[0]) / first_count
                        if monthly_counts[len(monthly_counts)//2] > 0:
                            second_half_growth = (monthly_counts[-1] - monthly_counts[len(monthly_counts)//2]) / monthly_counts[len(monthly_counts)//2]
                        else:
                            # If middle month has zero occurrences, use a different calculation
                            # Either use a small constant value, or calculate growth differently
                            if monthly_counts[-1] > 0:
                                second_half_growth = 1.0  # Indicate some positive growth since we went from 0 to something
                            else:
                                second_half_growth = 0.0  # No growth if both values are 0
                        acceleration = second_half_growth - first_half_growth
                    else:
                        acceleration = 0
                    
                    skill_growth.append({
                        'Skill': skill,
                        'Growth Rate': growth_rate,
                        'Acceleration': acceleration,
                        'Current Count': last_count,
                        'Monthly Counts': monthly_counts,
                        'Months': [str(m) for m in all_months]
                    })
            
            # Sort by a combination of growth and acceleration
            for skill in skill_growth:
                skill['Trend Score'] = skill['Growth Rate'] * 0.7 + skill['Acceleration'] * 0.3
            
            # Sort by trend score
            skill_growth.sort(key=lambda x: x['Trend Score'], reverse=True)
            emerging_skills = skill_growth[:15]  # Top 15 emerging skills
            
            # Visualize emerging skills
            st.markdown("### 🔮 Emerging Skills")
            
            # Create DataFrame for visualization
            emerging_df = pd.DataFrame([{
                'Skill': s['Skill'],
                'Growth Rate': round(s['Growth Rate'] * 100, 1),
                'Current Demand': s['Current Count']
            } for s in emerging_skills])
            
            # Create bar chart of growth rates
            fig1 = px.bar(
                emerging_df,
                y='Skill',
                x='Growth Rate',
                title="Fastest Growing Skills (% Growth)",
                color='Growth Rate',
                color_continuous_scale='Viridis',
                text='Growth Rate'
            )
            fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create line charts for top emerging skills
            st.markdown("### Skill Growth Trajectories")
            
            for i, skill in enumerate(emerging_skills[:5]):  # Show top 5 skills
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=skill['Months'],
                    y=skill['Monthly Counts'],
                    mode='lines+markers',
                    name='Trend',
                    line=dict(color='green')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Growth Trajectory for {skill['Skill']} (Growth Rate: {round(skill['Growth Rate']*100, 1)}%)",
                    xaxis_title="Month",
                    yaxis_title="Occurrences in Job Listings"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a recommendation section
            st.markdown("## 💡 Strategic Recommendations")
            st.markdown("### Skills to Acquire in the Next 6 Months")
            
            for i, skill in enumerate(emerging_skills[:7]):
                st.markdown(f"{i+1}. **{skill['Skill']}** - Growing at {round(skill['Growth Rate']*100, 1)}% with strong future potential")
            
        else:
            st.warning("Not enough time-series data for detailed predictions. Add more historical data with different dates.")
    else:
        st.warning("Date information not available for trend prediction")

def main():
    """Main function for the predictions dashboard"""
    st.markdown('<h1 style="text-align: center;">Job Market Future Forecasts</h1>', unsafe_allow_html=True)
    st.write("Predictive analysis of job market trends and emerging skills")
    
    # Load data from all companies
    df = load_all_company_data(DATA_DIR)
    
    if df is not None:
        # Predict job category growth
        predict_job_category_growth(df)
        
        # Predict skill demand trends
        predict_skill_demand(df)
        
        # Display limitation notice
        st.info("""
        **Note on Predictions**: These forecasts are based on available historical data and current trends.
        For more accurate predictions, consider adding more historical job data spanning a longer time period.
        """)
        
    else:
        st.error("No data available. Please add company data files to the data directory.")

if __name__ == "__main__":
    main()