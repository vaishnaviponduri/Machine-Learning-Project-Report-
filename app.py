import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Retail Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("customer_shopping_data.csv")
    
    # Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['price', 'quantity'])
    df['age'] = df['age'].fillna(df['age'].median())
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df['total_sales'] = df['price'] * df['quantity']
    
    # Feature engineering
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day'] = df['invoice_date'].dt.day
    df['weekday'] = df['invoice_date'].dt.weekday
    
    customer_avg_price = df.groupby('customer_id')['price'].mean().rename("customer_avg_price")
    df = df.merge(customer_avg_price, on='customer_id', how='left')
    
    # Additional features for visualization
    df['age_segment'] = pd.cut(
        df['age'],
        bins=[0, 18, 30, 45, 60, 100],
        labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
    )
    df['day_of_week'] = df['invoice_date'].dt.day_name()
    
    return df

# Train model
@st.cache_resource
def train_models(df):
    categorical_features = ['gender', 'category', 'payment_method', 'shopping_mall']
    numeric_features = ['age', 'quantity', 'price', 'customer_avg_price', 'year', 'month', 'day', 'weekday']
    
    X = df[categorical_features + numeric_features]
    y = df['total_sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pca = PCA(n_components=0.95)
    
    # Random Forest
    rf_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('pca', pca),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    
    # XGBoost
    xgb_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('pca', pca),
        ('model', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    
    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    
    # Calculate metrics
    rf_metrics = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    }
    
    xgb_metrics = {
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'R2': r2_score(y_test, xgb_pred)
    }
    
    return rf_pipeline, xgb_pipeline, rf_metrics, xgb_metrics, X_test, y_test, rf_pred, xgb_pred

# Main app
def main():
    st.title(" Retail Sales Prediction System")
    st.markdown("### Machine Learning-Powered Sales Forecasting")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        " Dashboard",
        " Data Exploration",
        " Model Performance",
        " Make Prediction"
    ])
    
    if page == " Dashboard":
        show_dashboard(df)
    elif page == " Data Exploration":
        show_exploration(df)
    elif page == " Model Performance":
        show_model_performance(df)
    elif page == " Make Prediction":
        show_prediction(df)

def show_dashboard(df):
    st.header(" Sales Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${df['total_sales'].sum():,.2f}")
    with col2:
        st.metric("Total Transactions", f"{len(df):,}")
    with col3:
        st.metric("Average Sale", f"${df['total_sales'].mean():,.2f}")
    with col4:
        st.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        category_sales = df.groupby('category')['total_sales'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            labels={'x': 'Category', 'y': 'Total Sales ($)'},
            color=category_sales.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Gender")
        gender_sales = df.groupby('gender')['total_sales'].sum()
        fig = px.pie(
            values=gender_sales.values,
            names=gender_sales.index,
            hole=0.4,
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Payment Method")
        payment_sales = df.groupby('payment_method')['total_sales'].sum()
        fig = px.bar(
            x=payment_sales.index,
            y=payment_sales.values,
            labels={'x': 'Payment Method', 'y': 'Total Sales ($)'},
            color=payment_sales.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Age Segment")
        age_sales = df.groupby('age_segment', observed=True)['total_sales'].sum()
        fig = px.bar(
            x=age_sales.index,
            y=age_sales.values,
            labels={'x': 'Age Segment', 'y': 'Total Sales ($)'},
            color=age_sales.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_exploration(df):
    st.header(" Data Exploration")
    
    tab1, tab2, tab3 = st.tabs([" Data Overview", " Distributions", " Correlations"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Date Range:** {df['invoice_date'].min().date()} to {df['invoice_date'].max().date()}")
        
        with col2:
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("No missing values!")
            else:
                st.dataframe(missing[missing > 0])
    
    with tab2:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['total_sales'].hist(bins=50, ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Total Sales ($)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Total Sales')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['age'].hist(bins=30, ax=ax, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Customer Age')
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['quantity'].hist(bins=20, ax=ax, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Quantity')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Quantity')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['price'].hist(bins=50, ax=ax, color='gold', edgecolor='black')
            ax.set_xlabel('Price ($)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Price')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Correlation Analysis")
        numeric_cols = ['age', 'quantity', 'price', 'total_sales', 'customer_avg_price']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

def show_model_performance(df):
    st.header(" Model Performance")
    
    with st.spinner("Training models... This may take a moment."):
        rf_model, xgb_model, rf_metrics, xgb_metrics, X_test, y_test, rf_pred, xgb_pred = train_models(df)
    
    st.success(" Models trained successfully!")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Random Forest")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("MAE", f"${rf_metrics['MAE']:,.2f}")
        with metric_col2:
            st.metric("RMSE", f"${rf_metrics['RMSE']:,.2f}")
        with metric_col3:
            st.metric("R² Score", f"{rf_metrics['R2']:.4f}")
    
    with col2:
        st.markdown("###  XGBoost")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("MAE", f"${xgb_metrics['MAE']:,.2f}")
        with metric_col2:
            st.metric("RMSE", f"${xgb_metrics['RMSE']:,.2f}")
        with metric_col3:
            st.metric("R² Score", f"{xgb_metrics['R2']:.4f}")
    
    st.markdown("---")
    
    # Prediction plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest: Actual vs Predicted")
        fig = px.scatter(
            x=y_test,
            y=rf_pred,
            labels={'x': 'Actual Sales ($)', 'y': 'Predicted Sales ($)'},
            opacity=0.6
        )
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("XGBoost: Actual vs Predicted")
        fig = px.scatter(
            x=y_test,
            y=xgb_pred,
            labels={'x': 'Actual Sales ($)', 'y': 'Predicted Sales ($)'},
            opacity=0.6,
            color_discrete_sequence=['#00CC96']
        )
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual plots
    st.subheader("Residual Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        rf_residuals = y_test - rf_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(rf_pred, rf_residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Sales ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Random Forest Residuals')
        st.pyplot(fig)
    
    with col2:
        xgb_residuals = y_test - xgb_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(xgb_pred, xgb_residuals, alpha=0.5, color='green')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Sales ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('XGBoost Residuals')
        st.pyplot(fig)

def show_prediction(df):
    st.header(" Make a Sales Prediction")
    
    st.markdown("### Enter Transaction Details")
    
    # Load models
    with st.spinner("Loading models..."):
        rf_model, xgb_model, _, _, _, _, _, _ = train_models(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", df['gender'].unique())
        category = st.selectbox("Category", df['category'].unique())
        payment_method = st.selectbox("Payment Method", df['payment_method'].unique())
    
    with col2:
        shopping_mall = st.selectbox("Shopping Mall", df['shopping_mall'].unique())
        age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
        quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
    
    with col3:
        price = st.number_input("Price per Unit ($)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
        year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
        month = st.selectbox("Month", range(1, 13))
    
    col1, col2 = st.columns(2)
    with col1:
        day = st.slider("Day", 1, 31, 15)

    with col2:
        weekday_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday = st.selectbox("Day of Week", weekday_options)
    # Convert selected weekday name to number 0–6
    weekday_num = weekday_options.index(weekday)
    
    # Calculate customer average price (use overall average as default)
    customer_avg_price = df['customer_avg_price'].mean()
    
    if st.button(" Predict Sales", type="primary", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'gender': [gender],
            'category': [category],
            'payment_method': [payment_method],
            'shopping_mall': [shopping_mall],
            'age': [age],
            'quantity': [quantity],
            'price': [price],
            'customer_avg_price': [customer_avg_price],
            'year': [year],
            'month': [month],
            'day': [day],
            'weekday': [weekday_num]
        })
        
        # Make predictions
        rf_prediction = rf_model.predict(input_data)[0]
        xgb_prediction = xgb_model.predict(input_data)[0]
        avg_prediction = (rf_prediction + xgb_prediction) / 2
        
        st.markdown("---")
        st.subheader(" Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(" Random Forest", f"${rf_prediction:,.2f}")
        with col2:
            st.metric(" XGBoost", f"${xgb_prediction:,.2f}")
        with col3:
            st.metric(" Average Prediction", f"${avg_prediction:,.2f}", 
                     delta=f"±${abs(rf_prediction - xgb_prediction)/2:,.2f}")
        
        st.success(f" Expected total sales: **${avg_prediction:,.2f}**")
        
        # Show calculation
        st.info(f" Base calculation: {quantity} × ${price:.2f} = ${quantity * price:.2f}")

if __name__ == "__main__":
    main()
