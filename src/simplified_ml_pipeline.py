import pandas as pd
import numpy as np
import boto3
import json
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO, StringIO
import holidays
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FixedPipelineConfig:
    """Configuration that creates missing buckets automatically"""
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
            self.account_id = "131201904254"
        
        # S3 bucket names
        self.raw_data_bucket = f"{self.project_name}-raw-data-{self.account_id}"
        self.processed_data_bucket = f"{self.project_name}-processed-data-{self.account_id}"
        self.features_bucket = f"{self.project_name}-features-{self.account_id}"
        self.models_bucket = f"{self.project_name}-models-{self.account_id}"
        self.predictions_bucket = f"{self.project_name}-predictions-{self.account_id}"
        
        # IAM role ARN
        self.sagemaker_role_arn = f"arn:aws:iam::{self.account_id}:role/{self.project_name}-sagemaker-execution-role"

class FixedMLPipeline:
    """
    ML Pipeline that automatically creates missing S3 buckets
    """
    
    def __init__(self, config: FixedPipelineConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.logger = self._setup_logging()
        self.uk_holidays = holidays.UK()
        
        # Model storage
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Ensure all buckets exist
        self._ensure_buckets_exist()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _ensure_buckets_exist(self):
        """Create missing S3 buckets"""
        buckets_to_check = {
            'raw_data': self.config.raw_data_bucket,
            'processed_data': self.config.processed_data_bucket,
            'features': self.config.features_bucket,
            'models': self.config.models_bucket,
            'predictions': self.config.predictions_bucket
        }
        
        for bucket_type, bucket_name in buckets_to_check.items():
            try:
                # Check if bucket exists
                self.s3_client.head_bucket(Bucket=bucket_name)
                self.logger.info(f"✅ {bucket_type} bucket exists: {bucket_name}")
            except Exception as e:
                if 'NoSuchBucket' in str(e) or '404' in str(e):
                    # Create the bucket
                    try:
                        if self.config.aws_region == 'us-east-1':
                            # us-east-1 doesn't need location constraint
                            self.s3_client.create_bucket(Bucket=bucket_name)
                        else:
                            # Other regions need location constraint
                            self.s3_client.create_bucket(
                                Bucket=bucket_name,
                                CreateBucketConfiguration={'LocationConstraint': self.config.aws_region}
                            )
                        self.logger.info(f"🆕 Created {bucket_type} bucket: {bucket_name}")
                    except Exception as create_error:
                        self.logger.error(f"❌ Failed to create {bucket_type} bucket: {create_error}")
                else:
                    self.logger.error(f"❌ Error checking {bucket_type} bucket: {e}")
    
    def load_data_from_s3(self) -> Dict[str, pd.DataFrame]:
        """Load data from your S3 bucket structure"""
        datasets = {}
        
        try:
            self.logger.info(f"🔍 Loading data from bucket: {self.config.raw_data_bucket}")
            
            # Get list of folders
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.raw_data_bucket,
                Delimiter='/'
            )
            
            if 'CommonPrefixes' in response:
                folder_names = [prefix['Prefix'].rstrip('/') for prefix in response['CommonPrefixes']]
                self.logger.info(f"📁 Found folders: {folder_names}")
                
                # Load each type of data
                for folder in folder_names:
                    data = self._load_folder_data(folder)
                    if not data.empty:
                        datasets[folder] = data
                        
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            
        return datasets
    
    def _load_folder_data(self, folder_name: str) -> pd.DataFrame:
        """Load all files from a folder"""
        try:
            all_dataframes = []
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.raw_data_bucket,
                Prefix=f"{folder_name}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('.parquet'):
                        obj_response = self.s3_client.get_object(
                            Bucket=self.config.raw_data_bucket,
                            Key=key
                        )
                        df = pd.read_parquet(BytesIO(obj_response['Body'].read()))
                        all_dataframes.append(df)
                    elif key.endswith('.csv'):
                        obj_response = self.s3_client.get_object(
                            Bucket=self.config.raw_data_bucket,
                            Key=key
                        )
                        df = pd.read_csv(BytesIO(obj_response['Body'].read()))
                        all_dataframes.append(df)
            
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                return combined_df.drop_duplicates()
            
        except Exception as e:
            self.logger.error(f"Failed to load folder {folder_name}: {str(e)}")
        
        return pd.DataFrame()
    
    def create_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive features for ML"""
        self.logger.info("🔧 Creating features for ML pipeline")
        
        # Start with energy data
        if 'neso_energy_data' in datasets:
            df = datasets['neso_energy_data'].copy()
        elif 'uk_energy_sample' in datasets:
            df = datasets['uk_energy_sample'].copy()
        else:
            raise ValueError("No energy data found")
        
        # Ensure datetime column
        datetime_col = self._find_datetime_column(df)
        if datetime_col:
            df['datetime'] = pd.to_datetime(df[datetime_col])
        else:
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
        
        # Find target column (demand)
        target_col = self._find_target_column(df)
        if target_col:
            df['target'] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Create temporal features
        df = self._create_temporal_features(df)
        
        # Merge weather data
        if 'uk_weather_current' in datasets or 'uk_weather' in datasets:
            weather_key = 'uk_weather_current' if 'uk_weather_current' in datasets else 'uk_weather'
            weather_df = datasets[weather_key]
            df = self._merge_weather_data(df, weather_df)
        
        # Merge carbon data
        if 'uk_carbon_intensity' in datasets:
            carbon_df = datasets['uk_carbon_intensity']
            df = self._merge_carbon_data(df, carbon_df)
        
        # Create lagged features
        df = self._create_lagged_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        self.logger.info(f"✅ Features created: {df.shape}")
        return df
    
    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column"""
        datetime_cols = ['datetime', 'timestamp', 'SETTLEMENT_DATE', 'time', 'date', 'datetime_from']
        for col in datetime_cols:
            if col in df.columns:
                return col
        return None
    
    def _find_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find target (demand) column"""
        target_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND', 'demand_mw', 'value', 'demand']
        for col in target_cols:
            if col in df.columns:
                return col
        return None
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (~df['is_weekend'].astype(bool))).astype(int)
        df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        
        # Seasonal features
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Holiday features
        df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: x in self.uk_holidays).astype(int)
        
        return df
    
    def _merge_weather_data(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data"""
        # Standardize weather column names
        weather_mapping = {
            'temperature_c': 'temperature',
            'humidity_percent': 'humidity',
            'pressure_hpa': 'pressure',
            'wind_speed_ms': 'wind_speed'
        }
        
        for old_name, new_name in weather_mapping.items():
            if old_name in weather_df.columns:
                weather_df[new_name] = weather_df[old_name]
        
        # Create weather features
        if 'temperature' in weather_df.columns:
            weather_df['hdd'] = np.maximum(18 - weather_df['temperature'], 0)
            weather_df['cdd'] = np.maximum(weather_df['temperature'] - 22, 0)
        
        # Take average weather values
        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'hdd', 'cdd']
        for feature in weather_features:
            if feature in weather_df.columns:
                df[feature] = weather_df[feature].mean()
        
        return df
    
    def _merge_carbon_data(self, df: pd.DataFrame, carbon_df: pd.DataFrame) -> pd.DataFrame:
        """Merge carbon intensity data"""
        carbon_features = ['carbon_intensity_actual', 'carbon_intensity_forecast']
        for feature in carbon_features:
            if feature in carbon_df.columns:
                df[feature] = carbon_df[feature].mean()
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        if 'target' in df.columns and len(df) > 24:
            # Lagged demand features
            df['target_lag_1h'] = df['target'].shift(1)
            df['target_lag_24h'] = df['target'].shift(24) if len(df) > 24 else df['target']
            df['target_lag_168h'] = df['target'].shift(168) if len(df) > 168 else df['target']
            
            # Rolling features
            df['target_ma_24h'] = df['target'].rolling(24, min_periods=1).mean()
            df['target_ma_168h'] = df['target'].rolling(168, min_periods=1).mean() if len(df) > 168 else df['target'].rolling(min(len(df), 24), min_periods=1).mean()
            
            # Ratio features
            df['target_to_daily_avg'] = df['target'] / (df['target_ma_24h'] + 1e-6)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Remove non-feature columns
        exclude_cols = ['datetime', 'target'] + [col for col in df.columns if col.startswith('SETTLEMENT') or col in ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y = df['target'].copy() if 'target' in df.columns else pd.Series([0] * len(df))
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = feature_cols
        self.logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models"""
        self.logger.info("🤖 Training ML models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross validation
            if name == 'linear_regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mae': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.logger.info(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        self.models['best'] = results[best_model_name]['model']
        self.models['best_name'] = best_model_name
        
        self.logger.info(f"✅ Best model: {best_model_name}")
        
        return results
    
    def save_models_to_s3(self, models_results: Dict[str, Any]) -> str:
        """Save trained models to S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save models
            for name, result in models_results.items():
                model_data = {
                    'model': result['model'],
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'metrics': {k: v for k, v in result.items() if k != 'model'},
                    'timestamp': timestamp
                }
                
                # Serialize model
                buffer = BytesIO()
                joblib.dump(model_data, buffer)
                
                # Upload to S3
                s3_key = f"models/{name}_model_{timestamp}.joblib"
                self.s3_client.put_object(
                    Bucket=self.config.models_bucket,
                    Key=s3_key,
                    Body=buffer.getvalue(),
                    ContentType='application/octet-stream'
                )
            
            # Save best model separately
            best_model_data = {
                'model': self.models['best'],
                'model_name': self.models['best_name'],
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'timestamp': timestamp
            }
            
            buffer = BytesIO()
            joblib.dump(best_model_data, buffer)
            
            s3_key = f"models/best_model_{timestamp}.joblib"
            self.s3_client.put_object(
                Bucket=self.config.models_bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            model_location = f"s3://{self.config.models_bucket}/models/best_model_{timestamp}.joblib"
            self.logger.info(f"✅ Models saved to S3: {model_location}")
            
            return model_location
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
            raise
    
    def load_model_from_s3(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Load model from S3"""
        try:
            if model_path is None:
                # Find latest model
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config.models_bucket,
                    Prefix="models/best_model_"
                )
                
                if 'Contents' not in response:
                    raise ValueError("No models found in S3")
                
                # Get latest model
                latest_obj = max(response['Contents'], key=lambda x: x['LastModified'])
                model_key = latest_obj['Key']
            else:
                model_key = model_path.replace(f"s3://{self.config.models_bucket}/", "")
            
            # Load model
            obj_response = self.s3_client.get_object(
                Bucket=self.config.models_bucket,
                Key=model_key
            )
            
            model_data = joblib.load(BytesIO(obj_response['Body'].read()))
            
            self.models['best'] = model_data['model']
            self.models['best_name'] = model_data.get('model_name', 'unknown')
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            self.logger.info(f"✅ Model loaded from S3: {model_key}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def make_predictions(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on test data"""
        if 'best' not in self.models:
            raise ValueError("No trained model available. Train or load a model first.")
        
        self.logger.info("🔮 Making predictions")
        
        # Prepare test data
        test_features = test_data[self.feature_columns].copy()
        
        # Handle missing values
        for col in self.feature_columns:
            if col not in test_features.columns:
                test_features[col] = 0
            test_features[col] = test_features[col].fillna(test_features[col].median())
        
        # Make predictions
        if self.models['best_name'] == 'linear_regression':
            test_features_scaled = self.scaler.transform(test_features)
            predictions = self.models['best'].predict(test_features_scaled)
        else:
            predictions = self.models['best'].predict(test_features)
        
        # Create results dataframe
        results = test_data.copy()
        results['predicted_demand'] = predictions
        results['prediction_timestamp'] = datetime.now()
        
        self.logger.info(f"✅ Predictions made for {len(results)} samples")
        
        return results
    
    def save_predictions_to_s3(self, predictions_df: pd.DataFrame) -> str:
        """Save predictions to S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"predictions/predictions_{timestamp}.parquet"
            
            # Save predictions
            buffer = BytesIO()
            predictions_df.to_parquet(buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.config.predictions_bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            location = f"s3://{self.config.predictions_bucket}/{s3_key}"
            self.logger.info(f"✅ Predictions saved to {location}")
            
            return location
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {str(e)}")
            raise
    
    def create_dashboard_data(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Create data for dashboard visualization"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions_df),
            'prediction_summary': {
                'mean_predicted_demand': float(predictions_df['predicted_demand'].mean()),
                'min_predicted_demand': float(predictions_df['predicted_demand'].min()),
                'max_predicted_demand': float(predictions_df['predicted_demand'].max()),
                'std_predicted_demand': float(predictions_df['predicted_demand'].std())
            }
        }
        
        # Add time series data for plotting
        if 'datetime' in predictions_df.columns:
            time_series = predictions_df[['datetime', 'predicted_demand']].copy()
            time_series['datetime'] = time_series['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            dashboard_data['time_series'] = time_series.to_dict('records')
        
        # Add hourly patterns
        if 'hour' in predictions_df.columns:
            hourly_avg = predictions_df.groupby('hour')['predicted_demand'].mean()
            dashboard_data['hourly_patterns'] = {
                'hours': hourly_avg.index.tolist(),
                'average_demand': hourly_avg.values.tolist()
            }
        
        # Add model performance (if actual values available)
        if 'target' in predictions_df.columns:
            actual_vs_pred = predictions_df[['target', 'predicted_demand']].dropna()
            if len(actual_vs_pred) > 0:
                mae = mean_absolute_error(actual_vs_pred['target'], actual_vs_pred['predicted_demand'])
                rmse = np.sqrt(mean_squared_error(actual_vs_pred['target'], actual_vs_pred['predicted_demand']))
                r2 = r2_score(actual_vs_pred['target'], actual_vs_pred['predicted_demand'])
                
                dashboard_data['model_performance'] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'model_name': self.models.get('best_name', 'unknown')
                }
        
        return dashboard_data
    
    def save_dashboard_data_to_s3(self, dashboard_data: Dict[str, Any]) -> str:
        """Save dashboard data to S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"dashboard/dashboard_data_{timestamp}.json"
            
            # Save dashboard data
            self.s3_client.put_object(
                Bucket=self.config.predictions_bucket,
                Key=s3_key,
                Body=json.dumps(dashboard_data, indent=2),
                ContentType='application/json'
            )
            
            # Also save as latest
            self.s3_client.put_object(
                Bucket=self.config.predictions_bucket,
                Key="dashboard/latest_dashboard_data.json",
                Body=json.dumps(dashboard_data, indent=2),
                ContentType='application/json'
            )
            
            location = f"s3://{self.config.predictions_bucket}/{s3_key}"
            self.logger.info(f"✅ Dashboard data saved to {location}")
            
            return location
            
        except Exception as e:
            self.logger.error(f"Failed to save dashboard data: {str(e)}")
            raise

def run_fixed_ml_pipeline():
    """Run the fixed ML pipeline that creates missing buckets"""
    print("🚀 FIXED ML PIPELINE (Auto-creates missing buckets)")
    print("=" * 60)
    
    config = FixedPipelineConfig()
    pipeline = FixedMLPipeline(config)
    
    try:
        # Step 1: Load data
        print("1️⃣ Loading data from S3...")
        datasets = pipeline.load_data_from_s3()
        
        if not datasets:
            print("❌ No data loaded")
            return
        
        print(f"✅ Loaded {len(datasets)} datasets")
        
        # Step 2: Create features
        print("\n2️⃣ Creating features...")
        features_df = pipeline.create_features(datasets)
        print(f"✅ Features created: {features_df.shape}")
        
        # Step 3: Prepare training data
        print("\n3️⃣ Preparing training data...")
        X, y = pipeline.prepare_training_data(features_df)
        print(f"✅ Training data prepared: {X.shape}")
        
        # Step 4: Train models
        print("\n4️⃣ Training models...")
        model_results = pipeline.train_models(X, y)
        print("✅ Models trained")
        
        # Step 5: Save models
        print("\n5️⃣ Saving models to S3...")
        model_location = pipeline.save_models_to_s3(model_results)
        print(f"✅ Models saved: {model_location}")
        
        # Step 6: Test predictions
        print("\n6️⃣ Testing predictions...")
        test_data = features_df.tail(100).copy()  # Use last 100 rows as test
        predictions = pipeline.make_predictions(test_data)
        print(f"✅ Predictions made: {len(predictions)} samples")
        
        # Step 7: Save predictions
        print("\n7️⃣ Saving predictions...")
        pred_location = pipeline.save_predictions_to_s3(predictions)
        print(f"✅ Predictions saved: {pred_location}")
        
        # Step 8: Create dashboard data
        print("\n8️⃣ Creating dashboard data...")
        dashboard_data = pipeline.create_dashboard_data(predictions)
        dashboard_location = pipeline.save_dashboard_data_to_s3(dashboard_data)
        print(f"✅ Dashboard data saved: {dashboard_location}")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"📊 Features: {features_df.shape[1]} features, {features_df.shape[0]} samples")
        print(f"🤖 Best model: {pipeline.models['best_name']}")
        print(f"🔮 Predictions: {len(predictions)} samples")
        print(f"📈 Dashboard ready for visualization")
        
        # Show sample predictions
        print(f"\n📋 Sample Predictions:")
        sample_cols = ['datetime', 'predicted_demand']
        if 'target' in predictions.columns:
            sample_cols.append('target')
        
        available_cols = [col for col in sample_cols if col in predictions.columns]
        if available_cols:
            print(predictions[available_cols].head(10))
        
        # Show model performance summary
        print(f"\n🎯 Model Performance Summary:")
        for name, result in model_results.items():
            print(f"  {name:15}: MAE={result['mae']:6.2f}, RMSE={result['rmse']:6.2f}, R²={result['r2']:5.3f}")
        
        return {
            'features_shape': features_df.shape,
            'best_model': pipeline.models['best_name'],
            'predictions_count': len(predictions),
            'model_location': model_location,
            'predictions_location': pred_location,
            'dashboard_location': dashboard_location,
            'model_results': model_results
        }
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Quick test function
def test_model_predictions():
    """Quick test of model predictions with sample data"""
    print("🧪 TESTING MODEL PREDICTIONS")
    print("=" * 40)
    
    config = FixedPipelineConfig()
    pipeline = FixedMLPipeline(config)
    
    try:
        # Load the trained model
        print("📥 Loading trained model...")
        pipeline.load_model_from_s3()
        print(f"✅ Loaded model: {pipeline.models['best_name']}")
        
        # Create test scenarios
        test_scenarios = {
            'Normal Weekday': {
                'hour': 14, 'day_of_week': 1, 'month': 6, 'is_weekend': 0, 'is_holiday': 0,
                'temperature': 20, 'humidity': 65, 'pressure': 1013, 'wind_speed': 8
            },
            'Weekend': {
                'hour': 14, 'day_of_week': 6, 'month': 6, 'is_weekend': 1, 'is_holiday': 0,
                'temperature': 22, 'humidity': 60, 'pressure': 1015, 'wind_speed': 5
            },
            'Cold Winter': {
                'hour': 18, 'day_of_week': 2, 'month': 1, 'is_weekend': 0, 'is_holiday': 0,
                'temperature': 2, 'humidity': 80, 'pressure': 1010, 'wind_speed': 12
            },
            'Hot Summer': {
                'hour': 15, 'day_of_week': 3, 'month': 7, 'is_weekend': 0, 'is_holiday': 0,
                'temperature': 32, 'humidity': 45, 'pressure': 1018, 'wind_speed': 3
            }
        }
        
        print("\n🔮 Making predictions for test scenarios:")
        print("-" * 50)
        
        for scenario_name, scenario_data in test_scenarios.items():
            # Create complete feature set
            test_data = pd.DataFrame([scenario_data])
            
            # Add derived features
            test_data['hour_sin'] = np.sin(2 * np.pi * test_data['hour'] / 24)
            test_data['hour_cos'] = np.cos(2 * np.pi * test_data['hour'] / 24)
            test_data['dow_sin'] = np.sin(2 * np.pi * test_data['day_of_week'] / 7)
            test_data['dow_cos'] = np.cos(2 * np.pi * test_data['day_of_week'] / 7)
            test_data['month_sin'] = np.sin(2 * np.pi * test_data['month'] / 12)
            test_data['month_cos'] = np.cos(2 * np.pi * test_data['month'] / 12)
            test_data['is_business_hours'] = 1 if 9 <= scenario_data['hour'] <= 17 and not scenario_data['is_weekend'] else 0
            test_data['is_peak_morning'] = 1 if 7 <= scenario_data['hour'] <= 9 else 0
            test_data['is_peak_evening'] = 1 if 17 <= scenario_data['hour'] <= 20 else 0
            test_data['is_winter'] = 1 if scenario_data['month'] in [12, 1, 2] else 0
            test_data['is_summer'] = 1 if scenario_data['month'] in [6, 7, 8] else 0
            test_data['hdd'] = max(18 - scenario_data['temperature'], 0)
            test_data['cdd'] = max(scenario_data['temperature'] - 22, 0)
            
            # Add missing features with default values
            for feature in pipeline.feature_columns:
                if feature not in test_data.columns:
                    test_data[feature] = 0
            
            # Make prediction
            prediction = pipeline.make_predictions(test_data)
            predicted_demand = prediction['predicted_demand'].iloc[0]
            
            print(f"{scenario_name:15}: {predicted_demand:6.0f} MW")
        
        print("\n✅ All predictions completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        test_model_predictions()
    else:
        # Full pipeline
        run_fixed_ml_pipeline()