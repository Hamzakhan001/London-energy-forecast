import pandas as pd
import numpy as np
import boto3
import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO
import holidays
import os

# Import SageMaker dependencies (optional)
try:
    import sagemaker
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.feature_store.feature_group import FeatureGroup
    from sagemaker.feature_store.feature_definition import FeatureDefinition
    from sagemaker.feature_store.inputs import FeatureTypeEnum
    from sagemaker import get_execution_role
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False
    print("⚠️ SageMaker not available. Some features will be limited.")

@dataclass
class PipelineConfig:
    """Configuration that works with your actual S3 bucket structure"""
    aws_region: str = "eu-west-1"
    environment: str = "dev"
    project_name: str = "energy-ml"
    
    def __post_init__(self):
        try:
            # Get AWS account ID
            sts_client = boto3.client('sts', region_name=self.aws_region)
            self.account_id = sts_client.get_caller_identity()['Account']
            print(f"✅ AWS Account ID: {self.account_id}")
        except Exception as e:
            print(f"⚠️ Could not get AWS account ID: {e}")
            self.account_id = "131201904254"  # Your actual account ID
        
        # Use your actual S3 bucket names
        self.raw_data_bucket = f"{self.project_name}-raw-data-{self.account_id}"
        self.processed_data_bucket = f"{self.project_name}-processed-data-{self.account_id}"
        self.features_bucket = f"{self.project_name}-features-{self.account_id}"
        self.models_bucket = f"{self.project_name}-models-{self.account_id}"
        self.predictions_bucket = f"{self.project_name}-predictions-{self.account_id}"
        
        # Kinesis stream names (from Terraform)
        self.kinesis_raw_stream = f"{self.project_name}-raw-data-stream"
        self.kinesis_processed_stream = f"{self.project_name}-processed-data-stream"
        self.kinesis_predictions_stream = f"{self.project_name}-predictions-stream"
        
        # IAM role ARN (from Terraform)
        self.sagemaker_role_arn = f"arn:aws:iam::{self.account_id}:role/{self.project_name}-sagemaker-execution-role"
        
        # Feature Store settings
        self.feature_store_database = "energy_feature_store"
        self.enable_streaming = True

class WorkingFeatureEngineering:
    """
    Feature engineering pipeline that works with your actual S3 structure
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        try:
            self.kinesis_client = boto3.client('kinesis', region_name=config.aws_region)
        except Exception as e:
            print(f"⚠️ Kinesis client not available: {e}")
            self.kinesis_client = None
            
        self.logger = self._setup_logging()
        
        # UK holidays
        self.uk_holidays = holidays.UK()
        
        # Complete feature categories for streaming
        self.feature_categories = {
            'temporal': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'],
            'weather': ['temperature', 'humidity', 'pressure', 'wind_speed', 'hdd', 'cdd', 'heat_index', 'wind_chill'],
            'demand': ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h', 'demand_ma_24h', 'demand_ma_168h'],
            'renewable': ['wind_cf', 'solar_cf', 'renewable_share', 'total_renewable_generation'],
            'calendar': ['is_weekend', 'is_holiday', 'days_to_holiday', 'is_business_hours', 'is_school_term'],
            'realtime': ['temp_rate_1h', 'demand_rate_1h', 'pressure_trend_6h', 'temp_anomaly_score', 'demand_anomaly_score'],
            'carbon': ['carbon_intensity_actual', 'carbon_intensity_forecast', 'carbon_forecast_error']
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_data_from_your_s3_structure(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from your actual S3 bucket structure
        Based on the folders I can see: neso_energy_data, uk_carbon_intensity, uk_energy_sample, uk_weather_current, uk_weather
        """
        datasets = {}
        
        try:
            # List what's actually in your raw data bucket
            self.logger.info(f"🔍 Scanning bucket: {self.config.raw_data_bucket}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.raw_data_bucket,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' in response:
                folder_names = [prefix['Prefix'].rstrip('/') for prefix in response['CommonPrefixes']]
                self.logger.info(f"📁 Found folders: {folder_names}")
                
                # Load NESO energy data
                if 'neso_energy_data' in folder_names:
                    datasets['neso'] = self._load_folder_data('neso_energy_data')
                    
                # Load weather data
                for weather_folder in ['uk_weather_current', 'uk_weather']:
                    if weather_folder in folder_names:
                        weather_data = self._load_folder_data(weather_folder)
                        if not weather_data.empty:
                            datasets['weather'] = weather_data
                            break
                
                # Load carbon intensity data  
                if 'uk_carbon_intensity' in folder_names:
                    datasets['carbon'] = self._load_folder_data('uk_carbon_intensity')
                
                # Load sample energy data if available
                if 'uk_energy_sample' in folder_names:
                    sample_data = self._load_folder_data('uk_energy_sample')
                    if not sample_data.empty and 'neso' not in datasets:
                        datasets['neso'] = sample_data
                        
            else:
                self.logger.warning("No folder structure found in bucket")
                
        except Exception as e:
            self.logger.error(f"Failed to scan S3 bucket: {str(e)}")
            
        # Log what we found
        for dataset_name, df in datasets.items():
            if not df.empty:
                self.logger.info(f"✅ Loaded {dataset_name}: {df.shape[0]} records, {df.shape[1]} columns")
                self.logger.info(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            else:
                self.logger.warning(f"⚠️ {dataset_name} dataset is empty")
                
        return datasets
    
    def _load_folder_data(self, folder_name: str) -> pd.DataFrame:
        """Load all data files from a specific folder"""
        try:
            all_dataframes = []
            
            # List all objects in the folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.raw_data_bucket,
                Prefix=f"{folder_name}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('.parquet'):
                        try:
                            # Load parquet file
                            obj_response = self.s3_client.get_object(
                                Bucket=self.config.raw_data_bucket,
                                Key=key
                            )
                            df = pd.read_parquet(BytesIO(obj_response['Body'].read()))
                            all_dataframes.append(df)
                            self.logger.info(f"   📄 Loaded {key}: {df.shape}")
                        except Exception as e:
                            self.logger.warning(f"   ❌ Failed to load {key}: {str(e)}")
                    elif key.endswith('.csv'):
                        try:
                            # Load CSV file
                            obj_response = self.s3_client.get_object(
                                Bucket=self.config.raw_data_bucket,
                                Key=key
                            )
                            df = pd.read_csv(BytesIO(obj_response['Body'].read()))
                            all_dataframes.append(df)
                            self.logger.info(f"   📄 Loaded {key}: {df.shape}")
                        except Exception as e:
                            self.logger.warning(f"   ❌ Failed to load {key}: {str(e)}")
            
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                combined_df = combined_df.drop_duplicates()
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to load folder {folder_name}: {str(e)}")
            return pd.DataFrame()
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features with flexible datetime column detection"""
        df = df.copy()
        
        # Find datetime column - try multiple common names
        datetime_col = None
        possible_datetime_cols = ['datetime', 'timestamp', 'SETTLEMENT_DATE', 'time', 'date', 'datetime_from']
        
        for col in possible_datetime_cols:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            # If no datetime column found, create one from current time
            self.logger.warning("No datetime column found, creating synthetic timestamps")
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
            datetime_col = 'datetime'
        else:
            # Convert to datetime
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            if datetime_col != 'datetime':
                df['datetime'] = df[datetime_col]
        
        # Basic temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['quarter'] = df['datetime'].dt.quarter
        
        # Cyclical encoding (essential for ML models)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Binary indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Business and peak hours
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (~df['is_weekend'].astype(bool))).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Seasonal indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # UK holidays
        df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: x in self.uk_holidays).astype(int)
        
        self.logger.info(f"Created temporal features from column: {datetime_col}")
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather features with flexible column detection"""
        df = df.copy()
        
        # Map possible weather column names to standard names
        weather_mapping = {
            'temperature_c': 'temperature',
            'temp': 'temperature',
            'Temperature': 'temperature',
            'humidity_percent': 'humidity',
            'Humidity': 'humidity',
            'pressure_hpa': 'pressure',
            'Pressure': 'pressure',
            'wind_speed_ms': 'wind_speed',
            'WindSpeed': 'wind_speed',
            'wind_speed': 'wind_speed'
        }
        
        # Rename columns to standard names
        for old_name, new_name in weather_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Check what weather data we have
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if not available_weather:
            self.logger.warning("No weather columns found, skipping weather features")
            return df
        
        self.logger.info(f"Creating weather features from: {available_weather}")
        
        # Core weather features
        if 'temperature' in df.columns:
            df['hdd'] = np.maximum(18 - df['temperature'], 0)
            df['cdd'] = np.maximum(df['temperature'] - 22, 0)
            
            # Temperature categories
            df['temp_extreme_cold'] = (df['temperature'] < 0).astype(int)
            df['temp_extreme_hot'] = (df['temperature'] > 30).astype(int)
            df['temp_comfortable'] = ((df['temperature'] >= 18) & (df['temperature'] <= 22)).astype(int)
        
        # Comfort indices (if we have the required data)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] - 50) / 100 * df['temperature']
        
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = df['temperature'] - 0.5 * df['wind_speed']
            df['wind_chill'] = df['wind_chill'].where(df['temperature'] < 10, df['temperature'])
        
        # Weather categories
        if 'wind_speed' in df.columns:
            df['wind_calm'] = (df['wind_speed'] < 2).astype(int)
            df['wind_strong'] = (df['wind_speed'] > 15).astype(int)
        
        if 'humidity' in df.columns:
            df['humidity_low'] = (df['humidity'] < 30).astype(int)
            df['humidity_high'] = (df['humidity'] > 80).astype(int)
        
        if 'pressure' in df.columns:
            df['pressure_low'] = (df['pressure'] < 1000).astype(int)
            df['pressure_high'] = (df['pressure'] > 1020).astype(int)
        
        # Simple lagged features (only if we have enough data)
        if len(df) > 24:
            for col in available_weather:
                df[f'{col}_lag_1h'] = df[col].shift(1)
                df[f'{col}_lag_24h'] = df[col].shift(24) if len(df) > 24 else df[col]
        
        self.logger.info(f"Created weather features")
        return df
    
    def create_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demand features with flexible demand column detection"""
        df = df.copy()
        
        # Find the demand column - try multiple possible names
        demand_col = None
        possible_demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND', 'demand_mw', 'value', 'demand', 'Demand']
        
        for col in possible_demand_cols:
            if col in df.columns:
                demand_col = col
                break
        
        if demand_col is None:
            self.logger.warning("No demand column found, skipping demand features")
            return df
        
        self.logger.info(f"Creating demand features from column: {demand_col}")
        
        # Convert to numeric if not already
        df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
        
        # Basic demand statistics
        df['demand_current'] = df[demand_col]
        
        # Lagged demand features (only if we have enough data)
        if len(df) > 1:
            df['demand_lag_1h'] = df[demand_col].shift(1)
        if len(df) > 24:
            df['demand_lag_24h'] = df[demand_col].shift(24)
        if len(df) > 168:
            df['demand_lag_168h'] = df[demand_col].shift(168)
        
        # Rolling statistics (only if we have enough data)
        if len(df) > 24:
            df['demand_ma_24h'] = df[demand_col].rolling(24, min_periods=1).mean()
            df['demand_std_24h'] = df[demand_col].rolling(24, min_periods=1).std()
        
        if len(df) > 168:
            df['demand_ma_168h'] = df[demand_col].rolling(168, min_periods=1).mean()
        
        # Demand trends
        if len(df) > 1:
            df['demand_rate_1h'] = df[demand_col].diff(1)
        
        # Demand categories (if we have enough data for percentiles)
        if len(df) > 100:
            q25 = df[demand_col].quantile(0.25)
            q75 = df[demand_col].quantile(0.75)
            df['demand_low'] = (df[demand_col] < q25).astype(int)
            df['demand_high'] = (df[demand_col] > q75).astype(int)
        
        self.logger.info(f"Created demand features")
        return df
    
    def create_carbon_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create carbon intensity features"""
        df = df.copy()
        
        # Find carbon intensity columns
        carbon_cols = []
        possible_carbon_cols = ['carbon_intensity_actual', 'carbon_intensity_forecast', 'intensity_forecast', 'intensity_actual']
        
        for col in possible_carbon_cols:
            if col in df.columns:
                carbon_cols.append(col)
        
        if not carbon_cols:
            self.logger.warning("No carbon intensity columns found, skipping carbon features")
            return df
        
        self.logger.info(f"Creating carbon features from: {carbon_cols}")
        
        for col in carbon_cols:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Carbon intensity categories
            df[f'{col}_low'] = (df[col] < 200).astype(int)
            df[f'{col}_medium'] = ((df[col] >= 200) & (df[col] < 300)).astype(int)
            df[f'{col}_high'] = (df[col] >= 300).astype(int)
            
            # Carbon trends (if we have enough data)
            if len(df) > 1:
                df[f'{col}_rate_1h'] = df[col].diff(1)
        
        # Forecast accuracy (if both actual and forecast are available)
        if 'carbon_intensity_actual' in df.columns and 'carbon_intensity_forecast' in df.columns:
            df['carbon_forecast_error'] = df['carbon_intensity_actual'] - df['carbon_intensity_forecast']
            df['carbon_forecast_abs_error'] = np.abs(df['carbon_forecast_error'])
        
        self.logger.info(f"Created carbon features")
        return df
    
    def create_all_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create all features from the loaded datasets"""
        self.logger.info("🔧 Creating all features from loaded datasets")
        
        # Start with the main energy/demand dataset
        if 'neso' in datasets and not datasets['neso'].empty:
            df = datasets['neso'].copy()
            self.logger.info(f"Base dataset (NESO): {df.shape}")
        else:
            self.logger.error("No NESO/energy data available")
            return pd.DataFrame()
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Merge weather data if available
        if 'weather' in datasets and not datasets['weather'].empty:
            weather_df = datasets['weather'].copy()
            
            # Try to merge on datetime
            if 'datetime' in df.columns and 'datetime' in weather_df.columns:
                df = df.merge(weather_df, on='datetime', how='left', suffixes=('', '_weather'))
            else:
                # Just take the first weather record for all energy records
                weather_sample = weather_df.iloc[0]
                for col in weather_df.columns:
                    if col not in df.columns:
                        df[col] = weather_sample[col]
            
            df = self.create_weather_features(df)
        
        # Merge carbon data if available
        if 'carbon' in datasets and not datasets['carbon'].empty:
            carbon_df = datasets['carbon'].copy()
            
            # Try to merge on datetime
            if 'datetime' in df.columns and 'datetime' in carbon_df.columns:
                df = df.merge(carbon_df, on='datetime', how='left', suffixes=('', '_carbon'))
            else:
                # Just take average carbon intensity for all records
                for col in carbon_df.select_dtypes(include=[np.number]).columns:
                    if col not in df.columns:
                        df[col] = carbon_df[col].mean()
            
            df = self.create_carbon_features(df)
        
        # Create demand features
        df = self.create_demand_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        self.logger.info(f"✅ Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        df = df.copy()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Fill object columns with mode or unknown
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'unknown'
                df[col] = df[col].fillna(fill_val)
        
        return df
    
    def save_features_to_s3(self, df: pd.DataFrame) -> str:
        """Save processed features to S3"""
        try:
            # Generate S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"processed_features/features_{timestamp}.parquet"
            
            # Save to S3
            buffer = BytesIO()
            df.to_parquet(buffer, index=False, engine='pyarrow')
            
            self.s3_client.put_object(
                Bucket=self.config.processed_data_bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                Metadata={
                    'pipeline_version': '2.0',
                    'feature_count': str(len(df.columns)),
                    'record_count': str(len(df)),
                    'processing_date': datetime.now().isoformat()
                }
            )
            
            s3_location = f"s3://{self.config.processed_data_bucket}/{s3_key}"
            self.logger.info(f"✅ Features saved to {s3_location}")
            
            return s3_location
            
        except Exception as e:
            self.logger.error(f"Failed to save features to S3: {str(e)}")
            raise

def test_working_feature_engineering():
    """Test the working feature engineering pipeline"""
    print("🧪 TESTING WORKING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    config = PipelineConfig()
    pipeline = WorkingFeatureEngineering(config)
    
    try:
        # Step 1: Load data from your actual S3 structure
        print("1️⃣ Loading data from your S3 buckets...")
        datasets = pipeline.load_data_from_your_s3_structure()
        
        if not datasets:
            print("❌ No datasets loaded from S3")
            return False
        
        print(f"✅ Loaded {len(datasets)} datasets:")
        for name, df in datasets.items():
            print(f"   📊 {name}: {df.shape[0]} records, {df.shape[1]} columns")
        
        # Step 2: Create features
        print("\n2️⃣ Creating features...")
        features_df = pipeline.create_all_features(datasets)
        
        if features_df.empty:
            print("❌ No features generated")
            return False
        
        print(f"✅ Features created: {features_df.shape}")
        
        # Step 3: Show feature summary
        print("\n3️⃣ Feature Summary:")
        feature_types = {
            'temporal': [col for col in features_df.columns if any(x in col for x in ['hour', 'day', 'month', 'sin', 'cos', 'is_'])],
            'weather': [col for col in features_df.columns if any(x in col for x in ['temp', 'humidity', 'pressure', 'wind', 'hdd', 'cdd'])],
            'demand': [col for col in features_df.columns if 'demand' in col],
            'carbon': [col for col in features_df.columns if 'carbon' in col],
        }
        
        for feature_type, features in feature_types.items():
            if features:
                print(f"   {feature_type}: {len(features)} features")
        
        # Step 4: Save to S3
        print("\n4️⃣ Saving features to S3...")
        s3_location = pipeline.save_features_to_s3(features_df)
        print(f"✅ Saved to: {s3_location}")
        
        # Step 5: Show sample data
        print("\n5️⃣ Sample Features:")
        sample_cols = ['datetime', 'hour_sin', 'hour_cos', 'is_weekend']
        if 'temperature' in features_df.columns:
            sample_cols.append('temperature')
        if 'demand_current' in features_df.columns:
            sample_cols.append('demand_current')
        
        available_cols = [col for col in sample_cols if col in features_df.columns]
        if available_cols:
            print(features_df[available_cols].head())
        
        print(f"\n🎉 SUCCESS! Generated {len(features_df.columns)} features from {len(features_df)} records")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_working_feature_engineering()