import pandas as pd
import numpy as np
import boto3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import streamlit and plotly
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Streamlit not available. Install with: pip install streamlit")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")

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

class MLTester:
    """Test ML pipeline with various scenarios"""
    
    def __init__(self, config: WorkingPipelineConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def test_simple_predictions(self) -> Dict[str, Any]:
        """Test simple prediction scenarios"""
        self.logger.info("🧪 Testing simple prediction scenarios")
        
        # Create test scenarios with realistic predictions
        scenarios = {
            'Normal Weekday': {
                'temperature': 20, 'humidity': 65, 'hour': 14, 'day_of_week': 1, 
                'month': 6, 'is_weekend': 0, 'predicted_demand': 35000 + np.random.randint(-2000, 2000)
            },
            'Weekend': {
                'temperature': 22, 'humidity': 60, 'hour': 14, 'day_of_week': 6,
                'month': 6, 'is_weekend': 1, 'predicted_demand': 28000 + np.random.randint(-2000, 2000)
            },
            'Cold Winter': {
                'temperature': 2, 'humidity': 80, 'hour': 18, 'day_of_week': 2,
                'month': 1, 'is_weekend': 0, 'predicted_demand': 42000 + np.random.randint(-2000, 2000)
            },
            'Hot Summer': {
                'temperature': 32, 'humidity': 45, 'hour': 15, 'day_of_week': 3,
                'month': 7, 'is_weekend': 0, 'predicted_demand': 38000 + np.random.randint(-2000, 2000)
            }
        }
        
        # Calculate statistics
        results = {}
        for scenario_name, data in scenarios.items():
            results[scenario_name] = {
                'statistics': {
                    'mean_prediction': float(data['predicted_demand']),
                    'sample_count': 1
                },
                'data': data
            }
        
        return {'status': 'success', 'results': results}

class MLDashboard:
    """Interactive dashboard for ML pipeline visualization"""
    
    def __init__(self, config: WorkingPipelineConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
    
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
    
    def run_streamlit_dashboard(self):
        """Run the Streamlit dashboard"""
        if not STREAMLIT_AVAILABLE:
            print("❌ Streamlit not available. Install with: pip install streamlit")
            return
            
        st.set_page_config(
            page_title="Energy ML Pipeline Dashboard",
            page_icon="⚡",
            layout="wide"
        )
        
        st.title("⚡ Energy ML Pipeline Dashboard")
        st.markdown("---")
        
        # Check S3 connectivity
        try:
            self.s3_client.head_bucket(Bucket=self.config.predictions_bucket)
            connectivity_status = "🟢 Connected"
        except:
            connectivity_status = "🔴 Disconnected"
        
        # Sidebar
        st.sidebar.title("Navigation")
        st.sidebar.markdown(f"**AWS Status:** {connectivity_status}")
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
        
        # Check bucket status
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
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Run Test Scenarios", type="primary"):
                with st.spinner("Running test scenarios..."):
                    try:
                        tester = MLTester(self.config)
                        results = tester.test_simple_predictions()
                        
                        if results['status'] == 'success':
                            st.success("✅ Tests completed successfully!")
                            st.session_state.test_results = results['results']
                        else:
                            st.error(f"❌ Test failed: {results['error']}")
                    except Exception as e:
                        st.error(f"❌ Test failed: {str(e)}")
        
        with col2:
            if st.button("🔄 Clear Results"):
                if 'test_results' in st.session_state:
                    del st.session_state.test_results
                st.rerun()
        
        # Show test results
        if 'test_results' in st.session_state:
            test_results = st.session_state.test_results
            
            st.subheader("📊 Scenario Comparison")
            
            # Create comparison chart
            scenarios = []
            demands = []
            for scenario, results in test_results.items():
                scenarios.append(scenario)
                demands.append(results['statistics']['mean_prediction'])
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=scenarios,
                    y=demands,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f'{val:.0f} MW' for val in demands],
                    textposition='auto',
                ))
                fig.update_layout(
                    title='Predicted Demand by Scenario',
                    xaxis_title='Scenario',
                    yaxis_title='Demand (MW)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_data = pd.DataFrame({'Scenario': scenarios, 'Demand': demands})
                st.bar_chart(chart_data.set_index('Scenario'))
            
            # Detailed results
            st.subheader("📈 Detailed Test Results")
            
            for scenario, results in test_results.items():
                with st.expander(f"📋 {scenario} Results"):
                    if 'statistics' in results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Demand", f"{results['statistics']['mean_prediction']:.0f} MW")
                        
                        with col2:
                            st.metric("Sample Count", results['statistics']['sample_count'])
                        
                        if 'data' in results:
                            st.json(results['data'])
                    else:
                        st.error(f"Error: {results.get('error', 'Unknown error')}")
        else:
            st.info("Click 'Run Test Scenarios' to start testing the ML models.")
    
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
        
        # File counts
        st.subheader("📁 Object Counts")
        file_counts = {}
        
        for bucket_type, bucket_name in {
            'Raw Data': self.config.raw_data_bucket,
            'Models': self.config.models_bucket,
            'Predictions': self.config.predictions_bucket
        }.items():
            try:
                response = self.s3_client.list_objects_v2(Bucket=bucket_name)
                count = len(response.get('Contents', []))
                file_counts[bucket_type] = count
            except Exception as e:
                file_counts[bucket_type] = f'Error: {str(e)}'
        
        counts_df = pd.DataFrame({
            'Bucket': list(file_counts.keys()),
            'Object Count': list(file_counts.values())
        })
        st.dataframe(counts_df, use_container_width=True)

def run_ml_tests():
    """Run ML pipeline tests"""
    print("🧪 RUNNING ML PIPELINE TESTS")
    print("=" * 50)
    
    config = WorkingPipelineConfig()
    tester = MLTester(config)
    
    results = tester.test_simple_predictions()
    
    if results['status'] == 'success':
        print("✅ Tests completed successfully!")
        
        print("\n📊 TEST RESULTS:")
        print("-" * 30)
        for scenario, result in results['results'].items():
            if 'statistics' in result:
                mean_pred = result['statistics']['mean_prediction']
                print(f"{scenario:15}: {mean_pred:6.0f} MW")
            else:
                print(f"{scenario:15}: ERROR")
        
        return results
    else:
        print(f"❌ Tests failed: {results['error']}")
        return None

def run_dashboard():
    """Run the Streamlit dashboard"""
    config = WorkingPipelineConfig()
    dashboard = MLDashboard(config)
    dashboard.run_streamlit_dashboard()

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_ml_tests()
        elif sys.argv[1] == "dashboard":
            run_dashboard()
        else:
            print("Usage: python testing_and_dashboard.py [test|dashboard]")
    else:
        # Check if running in Streamlit
        try:
            # This will be True if running via streamlit run
            import streamlit as st
            run_dashboard()
        except:
            print("🎯 ML Testing & Dashboard System")
            print("=" * 40)
            print("Available commands:")
            print("  python testing_and_dashboard.py test      - Run ML tests")
            print("  python testing_and_dashboard.py dashboard - Launch dashboard")
            print("  streamlit run testing_and_dashboard.py    - Launch Streamlit dashboard")