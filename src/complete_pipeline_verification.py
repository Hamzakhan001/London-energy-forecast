import boto3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

class PipelineVerification:
    """
    Complete verification of your ML pipeline infrastructure
    """
    
    def __init__(self):
        self.aws_region = "eu-west-1"
        self.account_id = "131201904254"
        self.project_name = "energy-ml"
        
        try:
            self.s3_client = boto3.client('s3', region_name=self.aws_region)
            self.kinesis_client = boto3.client('kinesis', region_name=self.aws_region)
            self.lambda_client = boto3.client('lambda', region_name=self.aws_region)
            self.iam_client = boto3.client('iam', region_name=self.aws_region)
            print("✅ AWS clients initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize AWS clients: {e}")
        
        # Expected bucket names
        self.buckets = {
            'raw_data': f"{self.project_name}-raw-data-{self.account_id}",
            'processed_data': f"{self.project_name}-processed-data-{self.account_id}",
            'features': f"{self.project_name}-features-{self.account_id}",
            'models': f"{self.project_name}-models-{self.account_id}",
            'predictions': f"{self.project_name}-predictions-{self.account_id}"
        }
        
        # Expected Kinesis streams
        self.streams = {
            'raw_data': f"{self.project_name}-raw-data-stream",
            'processed_data': f"{self.project_name}-processed-data-stream",
            'predictions': f"{self.project_name}-predictions-stream"
        }
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def verify_s3_buckets(self) -> Dict[str, bool]:
        """Verify all S3 buckets exist and check their contents"""
        print("\n🪣 VERIFYING S3 BUCKETS")
        print("=" * 50)
        
        bucket_status = {}
        
        for bucket_type, bucket_name in self.buckets.items():
            try:
                # Check if bucket exists
                response = self.s3_client.head_bucket(Bucket=bucket_name)
                print(f"✅ {bucket_type}: {bucket_name} exists")
                
                # Check bucket contents
                try:
                    objects = self.s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
                    if 'Contents' in objects:
                        print(f"   📁 Contains {objects.get('KeyCount', 0)} objects")
                        
                        # Show some sample objects
                        for obj in objects['Contents'][:3]:
                            size_mb = obj['Size'] / (1024 * 1024)
                            print(f"      📄 {obj['Key']} ({size_mb:.2f} MB)")
                    else:
                        print(f"   📭 Empty bucket")
                except Exception as e:
                    print(f"   ⚠️ Could not list contents: {e}")
                
                bucket_status[bucket_type] = True
                
            except Exception as e:
                print(f"❌ {bucket_type}: {bucket_name} - {e}")
                bucket_status[bucket_type] = False
        
        return bucket_status
    
    def verify_kinesis_streams(self) -> Dict[str, bool]:
        """Verify all Kinesis streams exist and are active"""
        print("\n🌊 VERIFYING KINESIS STREAMS")
        print("=" * 50)
        
        stream_status = {}
        
        for stream_type, stream_name in self.streams.items():
            try:
                response = self.kinesis_client.describe_stream(StreamName=stream_name)
                stream_info = response['StreamDescription']
                
                status = stream_info['StreamStatus']
                shard_count = len(stream_info['Shards'])
                retention = stream_info['RetentionPeriodHours']
                
                print(f"✅ {stream_type}: {stream_name}")
                print(f"   📊 Status: {status}")
                print(f"   🔢 Shards: {shard_count}")
                print(f"   ⏰ Retention: {retention} hours")
                
                if status == 'ACTIVE':
                    stream_status[stream_type] = True
                else:
                    stream_status[stream_type] = False
                    print(f"   ⚠️ Stream not active!")
                
            except Exception as e:
                print(f"❌ {stream_type}: {stream_name} - {e}")
                stream_status[stream_type] = False
        
        return stream_status
    
    def verify_iam_roles(self) -> Dict[str, bool]:
        """Verify IAM roles exist"""
        print("\n🔐 VERIFYING IAM ROLES")
        print("=" * 50)
        
        role_status = {}
        expected_roles = [
            f"{self.project_name}-sagemaker-execution-role",
            f"{self.project_name}-lambda-execution-role",
            f"{self.project_name}-kinesis-role"
        ]
        
        for role_name in expected_roles:
            try:
                response = self.iam_client.get_role(RoleName=role_name)
                print(f"✅ Role exists: {role_name}")
                role_status[role_name] = True
            except Exception as e:
                print(f"❌ Role missing: {role_name} - {e}")
                role_status[role_name] = False
        
        return role_status
    
    def test_data_ingestion(self) -> bool:
        """Test if data ingestion from your pipeline is working"""
        print("\n📥 TESTING DATA INGESTION")
        print("=" * 50)
        
        try:
            # Check raw data bucket for recent data
            bucket_name = self.buckets['raw_data']
            
            # List recent objects
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=50
            )
            
            if 'Contents' not in response:
                print("❌ No data found in raw data bucket")
                return False
            
            # Check for data from the last 7 days
            recent_objects = []
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) > cutoff_date:
                    recent_objects.append(obj)
            
            if recent_objects:
                print(f"✅ Found {len(recent_objects)} recent objects")
                for obj in recent_objects[:5]:
                    print(f"   📄 {obj['Key']} (modified: {obj['LastModified']})")
                return True
            else:
                print("⚠️ No recent data found (last 7 days)")
                return False
                
        except Exception as e:
            print(f"❌ Data ingestion test failed: {e}")
            return False
    
    def test_feature_engineering(self) -> bool:
        """Test feature engineering pipeline"""
        print("\n🔧 TESTING FEATURE ENGINEERING")
        print("=" * 50)
        
        try:
            # Import the working feature engineering pipeline
            from feature_engineering_working import WorkingFeatureEngineering, PipelineConfig
            
            config = PipelineConfig()
            pipeline = WorkingFeatureEngineering(config)
            
            # Load data
            datasets = pipeline.load_data_from_your_s3_structure()
            
            if not datasets:
                print("❌ No datasets loaded for feature engineering")
                return False
            
            print(f"✅ Loaded {len(datasets)} datasets")
            
            # Create features
            features_df = pipeline.create_all_features(datasets)
            
            if features_df.empty:
                print("❌ No features generated")
                return False
            
            print(f"✅ Generated {len(features_df.columns)} features from {len(features_df)} records")
            return True
            
        except ImportError:
            print("❌ Feature engineering module not found")
            return False
        except Exception as e:
            print(f"❌ Feature engineering test failed: {e}")
            return False
    
    def test_kinesis_streaming(self) -> bool:
        """Test Kinesis streaming functionality"""
        print("\n🌊 TESTING KINESIS STREAMING")
        print("=" * 50)
        
        try:
            stream_name = self.streams['raw_data']
            
            # Create test record
            test_record = {
                'timestamp': datetime.now().isoformat(),
                'test_data': 'pipeline_verification',
                'temperature': 15.5,
                'demand': 35000,
                'source': 'verification_test'
            }
            
            # Put record to Kinesis
            response = self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=json.dumps(test_record),
                PartitionKey='test'
            )
            
            print(f"✅ Test record sent to {stream_name}")
            print(f"   📨 Sequence number: {response['SequenceNumber']}")
            print(f"   🎯 Shard ID: {response['ShardId']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Kinesis streaming test failed: {e}")
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test the complete end-to-end pipeline"""
        print("\n🔄 TESTING END-TO-END PIPELINE")
        print("=" * 50)
        
        try:
            # 1. Load data from S3
            from feature_engineering_working import WorkingFeatureEngineering, PipelineConfig
            
            config = PipelineConfig()
            pipeline = WorkingFeatureEngineering(config)
            datasets = pipeline.load_data_from_your_s3_structure()
            
            if not datasets:
                print("❌ Step 1 failed: No data loaded")
                return False
            print("✅ Step 1: Data loaded from S3")
            
            # 2. Create features
            features_df = pipeline.create_all_features(datasets)
            if features_df.empty:
                print("❌ Step 2 failed: No features created")
                return False
            print("✅ Step 2: Features created")
            
            # 3. Save to processed bucket
            s3_location = pipeline.save_features_to_s3(features_df)
            print(f"✅ Step 3: Features saved to {s3_location}")
            
            # 4. Stream sample to Kinesis
            if len(features_df) > 0:
                sample_data = features_df.head(1).to_dict('records')[0]
                
                # Convert to JSON-serializable format
                for key, value in sample_data.items():
                    if pd.isna(value):
                        sample_data[key] = None
                    elif hasattr(value, 'item'):  # numpy types
                        sample_data[key] = value.item()
                
                response = self.kinesis_client.put_record(
                    StreamName=self.streams['processed_data'],
                    Data=json.dumps(sample_data, default=str),
                    PartitionKey='features'
                )
                print("✅ Step 4: Sample features streamed to Kinesis")
            
            print("\n🎉 END-TO-END PIPELINE TEST SUCCESSFUL!")
            return True
            
        except Exception as e:
            print(f"❌ End-to-end pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_deployment_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive deployment status report"""
        print("\n📋 GENERATING DEPLOYMENT STATUS REPORT")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'account_id': self.account_id,
            'region': self.aws_region,
            'project_name': self.project_name
        }
        
        # Run all verification tests
        report['s3_buckets'] = self.verify_s3_buckets()
        report['kinesis_streams'] = self.verify_kinesis_streams()
        report['iam_roles'] = self.verify_iam_roles()
        report['data_ingestion'] = self.test_data_ingestion()
        report['feature_engineering'] = self.test_feature_engineering()
        report['kinesis_streaming'] = self.test_kinesis_streaming()
        report['end_to_end_pipeline'] = self.test_end_to_end_pipeline()
        
        # Calculate overall status
        all_tests = [
            all(report['s3_buckets'].values()) if report['s3_buckets'] else False,
            all(report['kinesis_streams'].values()) if report['kinesis_streams'] else False,
            report['data_ingestion'],
            report['feature_engineering'],
            report['kinesis_streaming'],
            report['end_to_end_pipeline']
        ]
        
        report['overall_status'] = 'READY' if all(all_tests) else 'PARTIAL'
        report['ready_for_production'] = all(all_tests)
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final deployment summary"""
        print("\n" + "=" * 60)
        print("🎯 FINAL DEPLOYMENT STATUS")
        print("=" * 60)
        
        status = report['overall_status']
        if status == 'READY':
            print("🟢 STATUS: READY FOR PRODUCTION")
            print("\n✅ All systems operational:")
            print("   📦 S3 buckets configured")
            print("   🌊 Kinesis streams active") 
            print("   📥 Data ingestion working")
            print("   🔧 Feature engineering ready")
            print("   🔄 End-to-end pipeline tested")
            
            print("\n🚀 NEXT STEPS:")
            print("   1. Start real-time data ingestion")
            print("   2. Deploy ML models to SageMaker")
            print("   3. Set up monitoring and alerts")
            print("   4. Scale to production workloads")
            
        else:
            print("🟡 STATUS: PARTIAL DEPLOYMENT")
            print("\n⚠️ Issues found:")
            
            if not all(report['s3_buckets'].values()):
                failed_buckets = [k for k, v in report['s3_buckets'].items() if not v]
                print(f"   📦 S3 buckets missing: {failed_buckets}")
            
            if not all(report['kinesis_streams'].values()):
                failed_streams = [k for k, v in report['kinesis_streams'].items() if not v]
                print(f"   🌊 Kinesis streams missing: {failed_streams}")
            
            if not report['data_ingestion']:
                print("   📥 Data ingestion issues")
            
            if not report['feature_engineering']:
                print("   🔧 Feature engineering issues")
            
            print("\n🔧 REMEDIATION STEPS:")
            print("   1. Check Terraform deployment")
            print("   2. Verify IAM permissions")
            print("   3. Run data ingestion pipeline")
            print("   4. Re-test components")
        
        print(f"\n📊 Deployment Score: {sum(all_tests) if 'all_tests' in locals() else 0}/6 components working")
        print(f"🕒 Report generated: {report['timestamp']}")

def run_complete_verification():
    """Run complete pipeline verification"""
    print("🔍 COMPLETE PIPELINE VERIFICATION")
    print("=" * 60)
    print("Checking your entire ML pipeline deployment...")
    
    verifier = PipelineVerification()
    
    try:
        # Generate comprehensive report
        report = verifier.generate_deployment_status_report()
        
        # Print final summary
        verifier.print_final_summary(report)
        
        # Save report to file
        with open('pipeline_deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: pipeline_deployment_report.json")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_complete_verification()