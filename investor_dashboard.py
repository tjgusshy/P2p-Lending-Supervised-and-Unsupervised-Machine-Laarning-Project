
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 
# Page Configuration
# 
st.set_page_config(
    page_title="P2P Investment Dashboard",
    page_icon="💰",
    layout="wide"
)

# 
# Data Loading
# 
@st.cache_data
def load_data():
    df = pd.read_csv("borrower_clusters_with_pd.csv")
    return df

df = load_data()

# 
# Header
# 
st.title("💰 P2P Lending Investment Dashboard")
st.markdown("**Investment Dashboard**")
st.divider()

# 
# Sidebar Filters
# 
st.sidebar.header("🎯 Investment Filters")

# Risk level filter
risk_levels = sorted(df['risk_level'].unique())
selected_risks = st.sidebar.multiselect(
    "Risk Levels", 
    risk_levels, 
    default=risk_levels
)

# Loan amount filter
min_loan, max_loan = int(df['loan_amount'].min()), int(df['loan_amount'].max())
loan_range = st.sidebar.slider(
    "Loan Amount Range ($)", 
    min_loan, max_loan, 
    (min_loan, max_loan)
)

# Interest rate filter
min_rate, max_rate = float(df['interest_rate'].min()), float(df['interest_rate'].max())
rate_range = st.sidebar.slider(
    "Interest Rate Range (%)", 
    min_rate, max_rate, 
    (min_rate, max_rate)
)

# Income filter
min_income, max_income = int(df['annual_income'].min()), int(df['annual_income'].max())
income_range = st.sidebar.slider(
    "Annual Income Range ($)", 
    min_income, max_income, 
    (min_income, max_income)
)

# PD Score filter
min_pd, max_pd = float(df['pd_score'].min()), float(df['pd_score'].max())
pd_range = st.sidebar.slider(
    "PD Score Range", 
    min_pd, max_pd, 
    (min_pd, max_pd),
    format="%.3f"
)

# Apply filters
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

# 
# Key Performance Indicators
# 
st.subheader("📊 Investment Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_loans = len(filtered_df)
    st.metric("Available Loans", f"{total_loans:,}")

with col2:
    avg_loan_amount = filtered_df['loan_amount'].mean()
    st.metric("Avg Loan Amount", f"${avg_loan_amount:,.0f}")

with col3:
    avg_interest = filtered_df['interest_rate'].mean()
    st.metric("Avg Interest Rate", f"{avg_interest:.2f}%")

with col4:
    avg_pd_score = filtered_df['pd_score'].mean()
    st.metric("Avg Default Risk", f"{avg_pd_score:.3f}")

with col5:
    total_investment_opportunity = filtered_df['loan_amount'].sum()
    st.metric("Total Investment Pool", f"${total_investment_opportunity/1000000:.1f}M")

st.divider()

# 
# Investment Analysis Charts
# 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution")
    risk_counts = filtered_df['risk_level'].value_counts()
    
    # Risk level colors
    colors = {'Low': '#28a745', 'High': '#fd7e14', 'Very High': '#dc3545'}
    
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Loans by Risk Level",
        color=risk_counts.index,
        color_discrete_map=colors
    )
    fig_risk.update_layout(height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("Return vs Risk")
    fig_scatter = px.scatter(
        filtered_df,
        x='pd_score',
        y='interest_rate',
        color='risk_level',
        size='loan_amount',
        title="Interest Rate vs Default Probability",
        labels={'pd_score': 'Default Probability', 'interest_rate': 'Interest Rate (%)'},
        color_discrete_map=colors
    )
    fig_scatter.update_layout(height=300)
    st.plotly_chart(fig_scatter, use_container_width=True)

# 
# Investment Recommendations
# 
st.subheader("💡 Investment Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Low Risk Loans")
    low_risk = filtered_df[filtered_df['risk_level'] == 'Low']
    if len(low_risk) > 0:
        st.metric("Count", len(low_risk))
        st.metric("Avg Interest", f"{low_risk['interest_rate'].mean():.2f}%")
        st.metric("Avg Default Risk", f"{low_risk['pd_score'].mean():.3f}")
        st.success("✅ Recommended for conservative investors")
    else:
        st.info("No low risk loans in current filter")

with col2:
    st.markdown("### 🟡 High Risk Loans")
    high_risk = filtered_df[filtered_df['risk_level'] == 'High']
    if len(high_risk) > 0:
        st.metric("Count", len(high_risk))
        st.metric("Avg Interest", f"{high_risk['interest_rate'].mean():.2f}%")
        st.metric("Avg Default Risk", f"{high_risk['pd_score'].mean():.3f}")
        st.warning("⚠️ Higher returns, moderate risk")
    else:
        st.info("No high risk loans in current filter")

with col3:
    st.markdown("### 🔴 Very High Risk Loans")
    very_high_risk = filtered_df[filtered_df['risk_level'] == 'Very High']
    if len(very_high_risk) > 0:
        st.metric("Count", len(very_high_risk))
        st.metric("Avg Interest", f"{very_high_risk['interest_rate'].mean():.2f}%")
        st.metric("Avg Default Risk", f"{very_high_risk['pd_score'].mean():.3f}")
        st.error("🚨 High returns, high risk - experienced investors only")
    else:
        st.info("No very high risk loans in current filter")

st.divider()

# 
# Portfolio Analysis
# 
st.subheader("📈 Portfolio Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Loan Amounts by Risk Level")
    fig_box = px.box(
        filtered_df,
        x='risk_level',
        y='loan_amount',
        color='risk_level',
        title="Loan Amount Distribution",
        color_discrete_map=colors
    )
    fig_box.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    st.markdown("#### Interest Rates by Risk Level")
    fig_violin = px.violin(
        filtered_df,
        x='risk_level',
        y='interest_rate',
        color='risk_level',
        title="Interest Rate Distribution",
        color_discrete_map=colors
    )
    fig_violin.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

# 
# Risk Metrics Table
# 
st.subheader("📋 Risk Metrics Summary")

risk_summary = filtered_df.groupby('risk_level').agg({
    'loan_amount': ['count', 'mean', 'sum'],
    'interest_rate': 'mean',
    'pd_score': 'mean',
    'annual_income': 'mean'
}).round(2)

risk_summary.columns = ['Count', 'Avg Loan ($)', 'Total Pool ($)', 'Avg Rate (%)', 'Avg PD Score', 'Avg Income ($)']
st.dataframe(risk_summary, use_container_width=True)

# 
# Investment Decision Support
# 
st.subheader("🎯 Investment Decision Guide")

if total_loans > 0:
    # Portfolio allocation suggestion
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

# 
# Top Opportunities
# 
st.subheader("🏆 Top Investment Opportunities")

# Best low risk loans (high interest, low PD)
if len(low_risk) > 0:
    st.markdown("#### Best Low Risk Loans")
    best_low_risk = low_risk.nlargest(5, 'interest_rate')[['loan_amount', 'interest_rate', 'pd_score', 'annual_income', 'homeownership']]
    st.dataframe(best_low_risk, use_container_width=True)

# Best high risk loans (highest interest rates)
if len(high_risk) > 0:
    st.markdown("#### Best High Risk Loans")
    best_high_risk = high_risk.nlargest(5, 'interest_rate')[['loan_amount', 'interest_rate', 'pd_score', 'annual_income', 'homeownership']]
    st.dataframe(best_high_risk, use_container_width=True)

st.caption("💰 P2P Investment Dashboard - Make informed lending decisions based on risk analysis")