import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# page Configuration
#  
st.set_page_config(
    page_title="P2P Investment Dashboard",
    page_icon="💰",
    layout="wide"
)

#  
# data Loading
#  
@st.cache_data
def load_data():
    # Ensure this file exists in your directory
    df = pd.read_csv("borrower_clusters_with_pd.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'borrower_clusters_with_pd.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

 
# header
#  
st.title("💰 P2P Lending Investment Dashboard")
st.markdown("**Investment Dashboard**")
st.divider()

#sidebar Filters
#  
st.sidebar.header("🎯 Investment Filters")

#risk level filter
risk_levels = sorted(df['risk_level'].unique())
selected_risks = st.sidebar.multiselect(
    "Risk Levels", 
    risk_levels, 
    default=risk_levels
)

#loan amount filter
min_loan, max_loan = int(df['loan_amount'].min()), int(df['loan_amount'].max())
loan_range = st.sidebar.slider(
    "Loan Amount Range ($)", 
    min_loan, max_loan, 
    (min_loan, max_loan)
)

#interest rate filter
min_rate, max_rate = float(df['interest_rate'].min()), float(df['interest_rate'].max())
rate_range = st.sidebar.slider(
    "Interest Rate Range (%)", 
    min_rate, max_rate, 
    (min_rate, max_rate)
)

# annual income filter
min_income, max_income = int(df['annual_income'].min()), int(df['annual_income'].max())
income_range = st.sidebar.slider(
    "Annual Income Range ($)", 
    min_income, max_income, 
    (min_income, max_income)
)

#pd Score filter
min_pd, max_pd = float(df['pd_score'].min()), float(df['pd_score'].max())
pd_range = st.sidebar.slider(
    "PD Score Range", 
    min_pd, max_pd, 
    (min_pd, max_pd),
    format="%.3f"
)

#apply filters
filtered_df = df[
    (df['risk_level'].isin(selected_risks)) &
    (df['loan_amount'] >= loan_range[0]) &
    (df['loan_amount'] <= loan_range[1]) &
    (df['interest_rate'] >= rate_range[0]) &
    (df['interest_rate'] <= rate_range[1]) &
    (df['annual_income'] >= income_range[0]) &
    (df['annual_income'] <= income_range[1]) &
    (df['pd_score'] >= pd_range[0]) &
    (df['pd_score'] <= pd_range[1])
]

#key Performance Indicators (ADDED ICONS HERE)
#  
st.subheader("📊 Investment Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_loans = len(filtered_df)
    st.metric("📂 Available Loans", f"{total_loans:,}")

with col2:
    avg_loan_amount = filtered_df['loan_amount'].mean()
    val_loan = f"${avg_loan_amount:,.0f}" if not pd.isna(avg_loan_amount) else "$0"
    st.metric("💵 Avg Loan Amount", val_loan)

with col3:
    avg_interest = filtered_df['interest_rate'].mean()
    val_interest = f"{avg_interest:.2f}%" if not pd.isna(avg_interest) else "0%"
    st.metric("📈 Avg Interest Rate", val_interest)

with col4:
    avg_pd_score = filtered_df['pd_score'].mean()
    val_pd = f"{avg_pd_score:.3f}" if not pd.isna(avg_pd_score) else "0"
    st.metric("⚠️ Avg Default Risk", val_pd)

with col5:
    total_investment_opportunity = filtered_df['loan_amount'].sum()
    st.metric("🏦 Total Investment Pool", f"${total_investment_opportunity/1000000:.1f}M")

st.divider()

#investment Analysis Charts
#  
col1, col2 = st.columns(2)

#define Colors for consistency
colors = {'Low': '#28a745', 'High': '#fd7e14', 'Very High': '#dc3545'}

with col1:
    st.subheader("Risk Distribution")
    risk_counts = filtered_df['risk_level'].value_counts()
    
    if not risk_counts.empty:
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Loans by Risk Level",
            color=risk_counts.index,
            color_discrete_map=colors
        )
        fig_risk.update_layout(height=300)
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No data available for Risk Distribution")

with col2:
    st.subheader("Return vs Risk")
    if not filtered_df.empty:
        fig_scatter = px.scatter(
            filtered_df,
            x='pd_score',
            y='interest_rate',
            color='risk_level',
            size='loan_amount',
            title="Interest Rate vs Default Probability",
            labels={'pd_score': 'Default Probability', 'interest_rate': 'Interest Rate (%)'},
            color_discrete_map=colors,
            hover_data=['annual_income']
        )
        fig_scatter.update_layout(height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No data available for Scatter Plot")

#investment recommendations  
st.subheader("💡 Investment Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Low Risk Loans")
    low_risk = filtered_df[filtered_df['risk_level'] == 'Low']
    if len(low_risk) > 0:
        st.metric("🔢 Count", len(low_risk))
        st.metric("📈 Avg Interest", f"{low_risk['interest_rate'].mean():.2f}%")
        st.metric("🛡️ Avg Default Risk", f"{low_risk['pd_score'].mean():.3f}")
        st.success("✅ Recommended for conservative investors")
    else:
        st.info("No low risk loans in current filter")

with col2:
    st.markdown("### 🟡 High Risk Loans")
    high_risk = filtered_df[filtered_df['risk_level'] == 'High']
    if len(high_risk) > 0:
        st.metric("🔢 Count", len(high_risk))
        st.metric("📈 Avg Interest", f"{high_risk['interest_rate'].mean():.2f}%")
        st.metric("⚖️ Avg Default Risk", f"{high_risk['pd_score'].mean():.3f}")
        st.warning("⚠️ Higher returns, moderate risk")
    else:
        st.info("No high risk loans in current filter")

with col3:
    st.markdown("### 🔴 Very High Risk Loans")
    very_high_risk = filtered_df[filtered_df['risk_level'] == 'Very High']
    if len(very_high_risk) > 0:
        st.metric("🔢 Count", len(very_high_risk))
        st.metric("📈 Avg Interest", f"{very_high_risk['interest_rate'].mean():.2f}%")
        st.metric("🔥 Avg Default Risk", f"{very_high_risk['pd_score'].mean():.3f}")
        st.error("🚨 High returns, high risk - experienced investors only")
    else:
        st.info("No very high risk loans in current filter")

st.divider()

#portfolio Analysis (Business Friendly Charts)
#  
st.subheader("📈 Portfolio Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Loan Amounts by Risk Level")
    if not filtered_df.empty:
        # STRIP PLOT: Better for showing "Volume" of loans to business users
        fig_strip = px.strip(
            filtered_df,
            x='risk_level',
            y='loan_amount',
            color='risk_level',
            title="Individual Loan Distribution",
            color_discrete_map=colors,
            stripmode='overlay'
        )
        fig_strip.update_traces(marker=dict(opacity=0.6, size=4))
        fig_strip.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_strip, use_container_width=True)

with col2:
    st.markdown("#### Interest Rates by Risk Level")
    if not filtered_df.empty:
        # BAR CHART: Better for showing "Average" to business users
        # Pre-calculating mean and std for the bar chart
        df_avg = filtered_df.groupby('risk_level')['interest_rate'].agg(['mean', 'std']).reset_index()
        
        fig_bar = px.bar(
            df_avg,
            x='risk_level',
            y='mean',
            error_y='std', # Adds error bars to show risk/variance
            color='risk_level',
            title="Avg Interest Rate (+/- Deviation)",
            color_discrete_map=colors,
            text_auto='.2f'
        )
        fig_bar.update_layout(height=350, showlegend=False, yaxis_title="Interest Rate (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

#risk metrics Table
#  
st.subheader("📋 Risk Metrics Summary")

if not filtered_df.empty:
    risk_summary = filtered_df.groupby('risk_level').agg({
        'loan_amount': ['count', 'mean', 'sum'],  
        'interest_rate': 'mean',                  
        'pd_score': 'mean',                       
        'annual_income': 'mean'                   
    }).round(2)
    
    # Corrected column names (6 names for 6 columns)
    risk_summary.columns = [
        'Count', 
        'Avg Loan ($)', 
        'Total Pool ($)', 
        'Avg Rate (%)', 
        'Avg PD Score', 
        'Avg Income ($)'
    ]
    
    st.dataframe(risk_summary, use_container_width=True)
else:
    st.warning("No data matches the current filters.")

#  
# investment decision Support
#  
st.subheader("🎯 Investment Decision Guide")

if total_loans > 0:
    low_risk_pct = len(low_risk) / total_loans * 100
    high_risk_pct = len(high_risk) / total_loans * 100
    very_high_risk_pct = len(very_high_risk) / total_loans * 100
    
    st.markdown("#### Portfolio Allocation Suggestions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Conservative Portfolio**")
        st.write(f"Low Risk: {min(low_risk_pct, 80):.1f}%")
        st.write(f"High Risk: {min(high_risk_pct, 20):.1f}%")
        st.write("Very High Risk: 0%")
    
    with col2:
        st.markdown("**Balanced Portfolio**")
        st.write(f"Low Risk: {min(low_risk_pct, 60):.1f}%")
        st.write(f"High Risk: {min(high_risk_pct, 30):.1f}%")
        st.write(f"Very High Risk: {min(very_high_risk_pct, 10):.1f}%")
    
    with col3:
        st.markdown("**Aggressive Portfolio**")
        st.write(f"Low Risk: {min(low_risk_pct, 40):.1f}%")
        st.write(f"High Risk: {min(high_risk_pct, 40):.1f}%")
        st.write(f"Very High Risk: {min(very_high_risk_pct, 20):.1f}%")

st.divider()

# top opportunities
#  
st.subheader("🏆 Top Investment Opportunities")

if len(low_risk) > 0:
    st.markdown("#### Best Low Risk Loans")
    # Using safe access to columns to prevent key errors
    cols_to_show = ['loan_amount', 'interest_rate', 'pd_score', 'annual_income']
    if 'homeownership' in low_risk.columns:
        cols_to_show.append('homeownership')
        
    best_low_risk = low_risk.nlargest(5, 'interest_rate')[cols_to_show]
    st.dataframe(best_low_risk, use_container_width=True)

if len(high_risk) > 0:
    st.markdown("#### Best High Risk Loans")
    cols_to_show = ['loan_amount', 'interest_rate', 'pd_score', 'annual_income']
    if 'homeownership' in high_risk.columns:
        cols_to_show.append('homeownership')

    best_high_risk = high_risk.nlargest(5, 'interest_rate')[cols_to_show]
    st.dataframe(best_high_risk, use_container_width=True)

st.caption("💰 P2P Investment Dashboard - Make informed lending decisions based on risk analysis")