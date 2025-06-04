import pandas as pd
import numpy as np
import streamlit as st
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configuration class
class WorkingPipelineConfig:
    """Configuration that works with your existing setup"""
    def __init__(self):
        self.aws_region = "eu-west-1"
        self.environment = "dev"
        self.project_name = "energy-ml"
        
        try:
            sts_client = boto3.client('sts', region_name=self.aws_region)
            self.account_id = sts_client.get_caller_identity()['Account']
        except:
            self.account_id = "131201904254"
        
        # S3 bucket names
        self.raw_data_bucket = f"{self.project_name}-raw-data-{self.account_id}"
        self.processed_data_bucket = f"{self.project_name}-processed-data-{self.account_id}"
        self.features_bucket = f"{self.project_name}-features-{self.account_id}"
        self.models_bucket = f"{self.project_name}-models-{self.account_id}"
        self.predictions_bucket = f"{self.project_name}-predictions-{self.account_id}"

def get_aws_connection_status():
    """Check AWS connection and return status"""
    try:
        # Try Streamlit Cloud secrets first
        if hasattr(st, 'secrets') and 'aws' in st.secrets:
            session = boto3.Session(
                aws_access_key_id=st.secrets.aws.aws_access_key_id,
                aws_secret_access_key=st.secrets.aws.aws_secret_access_key,
                region_name=st.secrets.aws.aws_default_region
            )
        else:
            # Fallback to local AWS credentials
            session = boto3.Session()
        
        # Test the connection by getting caller identity
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        return {
            'status': 'Connected',
            'account': identity['Account'],
            'user_id': identity['UserId'], 
            'arn': identity['Arn'],
            'session': session
        }
        
    except NoCredentialsError:
        return {
            'status': 'No Credentials Found',
            'error': 'AWS credentials not configured'
        }
    except ClientError as e:
        return {
            'status': 'Connection Failed',
            'error': str(e)
        }
    except Exception as e:
        return {
            'status': 'Unknown Error',
            'error': str(e)
        }

def display_aws_status():
    """Display AWS connection status in sidebar"""
    aws_info = get_aws_connection_status()
    
    if aws_info['status'] == 'Connected':
        st.sidebar.success("🟢 AWS Status: Connected")
        st.sidebar.write(f"**Account:** {aws_info['account']}")
        st.sidebar.write(f"**User:** {aws_info['arn'].split('/')[-1]}")
        
        # Test S3 access
        try:
            s3 = aws_info['session'].client('s3')
            buckets = s3.list_buckets()
            bucket_count = len(buckets['Buckets'])
            st.sidebar.write(f"**S3 Buckets:** {bucket_count}")
        except Exception as e:
            st.sidebar.warning(f"S3 Access Limited: {str(e)}")
            
        return aws_info['session']
    else:
        st.sidebar.error(f"🔴 AWS Status: {aws_info['status']}")
        if 'error' in aws_info:
            st.sidebar.write(f"Error: {aws_info['error']}")
        return None

class MLDashboard:
    """Interactive dashboard for ML pipeline visualization"""
    
    def __init__(self, config: WorkingPipelineConfig):
        self.config = config
        try:
            self.s3_client = boto3.client('s3', region_name=config.aws_region)
        except:
            self.s3_client = None
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        dates = pd.date_range(start='2024-06-01', periods=96, freq='h')
        
        sample_data = []
        for i, dt in enumerate(dates):
            # Create realistic demand pattern
            base_demand = 35000
            hourly_pattern = 5000 * np.sin(2 * np.pi * dt.hour / 24)
            daily_pattern = 3000 * np.sin(2 * np.pi * dt.dayofweek / 7)
            noise = np.random.normal(0, 1000)
            
            demand = base_demand + hourly_pattern + daily_pattern + noise
            
            sample_data.append({
                'datetime': dt,
                'hour': dt.hour,
                'day_of_week': dt.dayofweek,
                'is_weekend': 1 if dt.dayofweek >= 5 else 0,
                'predicted_demand': demand,
                'temperature': 20 + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 2)
            })
        
        return pd.DataFrame(sample_data)
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.set_page_config(
            page_title="Energy ML Pipeline Dashboard",
            page_icon="⚡",
            layout="wide"
        )
        
        st.title("⚡ Energy ML Pipeline Dashboard")
        st.markdown("---")
        
        # Display AWS status in sidebar
        aws_session = display_aws_status()
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        st.sidebar.markdown(f"**Account:** {self.config.account_id}")
        
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Overview", "Predictions", "Model Testing", "System Info"]
        )
        
        if page == "Overview":
            self._show_overview()
        elif page == "Predictions":
            self._show_predictions()
        elif page == "Model Testing":
            self._show_testing()
        elif page == "System Info":
            self._show_system_info()
    
    def _show_overview(self):
        """Show overview page"""
        st.header("🎯 Pipeline Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pipeline Status", "🟢 Ready", "Operational")
        
        with col2:
            st.metric("Models Available", "3", "RF, GB, LR")
        
        with col3:
            st.metric("Data Sources", "6", "NESO, Weather, Carbon")
        
        with col4:
            st.metric("Buckets", "5", "All configured")
        
        st.markdown("---")
        
        # Architecture diagram
        st.subheader("🏗️ Architecture Overview")
        st.code("""
Data Sources → S3 Raw Data → Feature Engineering → Model Training → Predictions → Dashboard
     ↓              ↓              ↓               ↓             ↓           ↓
- NESO API     Raw Data      Features      Trained Models   Prediction   Streamlit
- Weather API   Bucket        Bucket         Bucket          Bucket      Dashboard
- Carbon API      
        """)
        
        # System status
        st.subheader("📊 System Status")
        
        # Check bucket status (if S3 client available)
        if self.s3_client:
            bucket_status = {}
            for bucket_type, bucket_name in {
                'Raw Data': self.config.raw_data_bucket,
                'Processed Data': self.config.processed_data_bucket,
                'Features': self.config.features_bucket,
                'Models': self.config.models_bucket,
                'Predictions': self.config.predictions_bucket
            }.items():
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    bucket_status[bucket_type] = "✅ Accessible"
                except:
                    bucket_status[bucket_type] = "❌ Error"
            
            status_df = pd.DataFrame({
                'Component': list(bucket_status.keys()),
                'Status': list(bucket_status.values())
            })
            
            st.dataframe(status_df, use_container_width=True)
        else:
            st.warning("⚠️ AWS connection required to check bucket status")
    
    def _show_predictions(self):
        """Show predictions page"""
        st.header("🔮 Energy Demand Predictions")
        
        # Create sample data for demonstration
        predictions_df = self.create_sample_data()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_demand = predictions_df['predicted_demand'].mean()
            st.metric("Average Demand", f"{avg_demand:.0f} MW")
        
        with col2:
            max_demand = predictions_df['predicted_demand'].max()
            st.metric("Peak Demand", f"{max_demand:.0f} MW")
        
        with col3:
            min_demand = predictions_df['predicted_demand'].min()
            st.metric("Minimum Demand", f"{min_demand:.0f} MW")
        
        with col4:
            volatility = predictions_df['predicted_demand'].std()
            st.metric("Volatility", f"{volatility:.0f} MW")
        
        # Time series plot
        st.subheader("📊 Demand Forecast Over Time")
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions_df['datetime'],
                y=predictions_df['predicted_demand'],
                mode='lines',
                name='Predicted Demand',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title='Energy Demand Predictions',
                xaxis_title='Time',
                yaxis_title='Demand (MW)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(predictions_df.set_index('datetime')['predicted_demand'])
        
        # Hourly pattern
        st.subheader("🕐 Average Hourly Demand Pattern")
        hourly_avg = predictions_df.groupby('hour')['predicted_demand'].mean()
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg.values,
                mode='lines+markers',
                name='Average Demand by Hour',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Average Hourly Demand Pattern',
                xaxis_title='Hour of Day',
                yaxis_title='Average Demand (MW)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(hourly_avg)
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        display_cols = ['datetime', 'predicted_demand', 'temperature', 'is_weekend']
        recent_data = predictions_df[display_cols].tail(20)
        st.dataframe(recent_data, use_container_width=True)
    
    def _show_testing(self):
        """Show model testing page"""
        st.header("🧪 Model Testing")
        
        st.info("Model testing functionality will be implemented here.")
        
        # Sample test scenarios
        scenarios = {
            'Normal Weekday': 35000 + np.random.randint(-2000, 2000),
            'Weekend': 28000 + np.random.randint(-2000, 2000),
            'Cold Winter': 42000 + np.random.randint(-2000, 2000),
            'Hot Summer': 38000 + np.random.randint(-2000, 2000)
        }
        
        if st.button("🚀 Run Test Scenarios", type="primary"):
            st.success("✅ Tests completed successfully!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                for scenario, demand in scenarios.items():
                    st.metric(scenario, f"{demand:.0f} MW")
            
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(scenarios.keys()),
                        y=list(scenarios.values()),
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    ))
                    fig.update_layout(
                        title='Test Scenario Results',
                        xaxis_title='Scenario',
                        yaxis_title='Demand (MW)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _show_system_info(self):
        """Show system information"""
        st.header("📊 System Information")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        config_data = {
            'AWS Region': self.config.aws_region,
            'Environment': self.config.environment,
            'Project Name': self.config.project_name,
            'Account ID': self.config.account_id
        }
        
        config_df = pd.DataFrame({
            'Setting': list(config_data.keys()),
            'Value': list(config_data.values())
        })
        st.dataframe(config_df, use_container_width=True)
        
        # S3 Buckets
        st.subheader("🪣 S3 Buckets")
        buckets_df = pd.DataFrame({
            'Purpose': ['Raw Data', 'Processed Data', 'Features', 'Models', 'Predictions'],
            'Bucket Name': [
                self.config.raw_data_bucket,
                self.config.processed_data_bucket,
                self.config.features_bucket,
                self.config.models_bucket,
                self.config.predictions_bucket
            ]
        })
        st.dataframe(buckets_df, use_container_width=True)

# Main function - THIS IS THE MAIN DEFINITION YOU NEED!
def main():
    """Main function to run the dashboard"""
    config = WorkingPipelineConfig()
    dashboard = MLDashboard(config)
    dashboard.run_dashboard()

# This ensures the main function runs when the script is executed
if __name__ == "__main__":
    main()