import boto3
import pandas as pd
import json
import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass
from io import BytesIO
import time

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    aws_region: str = "eu-west-1"
    project_name: str = "energy-ml"
    
    def __post_init__(self):
        sts_client = boto3.client('sts', region_name=self.aws_region)
        self.account_id = sts_client.get_caller_identity()['Account']
        self.raw_data_bucket = f"{self.project_name}-raw-data-{self.account_id}"

class EfficientUKEnergyPipeline:
    """
    Efficient UK Energy Pipeline - Minimal API usage
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.logger = self._setup_logging()
        
        # API credentials from environment
        self.weather_api_key = os.getenv('WEATHER_API_KEY', '995cc31c53b437ac956f1fcab3c8baad')
        self.met_api_key = os.getenv('MET_API_KEY', '')
        
        # API endpoints
        self.neso_base_url = "https://api.neso.energy/api/3/action/datastore_search"
        self.elexon_base_url = "https://data.elexon.co.uk/bmrs/api/v1"  # Updated to v1 API
        self.carbon_intensity_url = "https://api.carbonintensity.org.uk"
        self.met_office_url = "https://api-metoffice.apiconnect.ibmcloud.com/metoffice/production"  # Updated Met Office endpoint
        
        self.logger.info("🔧 Pipeline initialized with real API credentials")
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def test_all_apis(self) -> Dict[str, Any]:
        """
        Test all APIs with single requests to verify connectivity
        """
        print("🧪 TESTING ALL APIs - Single requests only")
        print("=" * 60)
        
        results = {}
        
        # Test 1: NESO Energy API
        print("1️⃣ Testing NESO Energy API...")
        try:
            url = self.neso_base_url
            params = {
                'resource_id': 'f6d02c0f-957b-48cb-82ee-09003f2ba759',
                'limit': 5
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                record_count = len(data.get('result', {}).get('records', []))
                results['neso'] = {
                    'status': 'success',
                    'records_found': record_count,
                    'sample_data': data.get('result', {}).get('records', [])[:2] if record_count > 0 else []
                }
                print(f"   ✅ SUCCESS: {record_count} records found")
            else:
                results['neso'] = {'status': 'failed', 'error': f"HTTP {response.status_code}"}
                print(f"   ❌ FAILED: HTTP {response.status_code}")
                
        except Exception as e:
            results['neso'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ FAILED: {str(e)}")
        
        time.sleep(1)  # Rate limiting for NESO
        
        # Test 2: OpenWeatherMap API
        print("\n2️⃣ Testing OpenWeatherMap API...")
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': 'London,UK',
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results['openweather'] = {
                    'status': 'success',
                    'city': data.get('name'),
                    'temperature': data.get('main', {}).get('temp'),
                    'description': data.get('weather', [{}])[0].get('description', '')
                }
                print(f"   ✅ SUCCESS: {data.get('name')} - {data.get('main', {}).get('temp')}°C")
            else:
                results['openweather'] = {'status': 'failed', 'error': f"HTTP {response.status_code}"}
                print(f"   ❌ FAILED: HTTP {response.status_code}")
                
        except Exception as e:
            results['openweather'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ FAILED: {str(e)}")
        
        # Test 3: Elexon BMRS API (new v1 API)
        print("\n3️⃣ Testing Elexon BMRS API...")
        try:
            # Use the new BMRS v1 API - try a simple system-wide endpoint
            url = "https://data.elexon.co.uk/bmrs/api/v1/datasets"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results['elexon'] = {
                    'status': 'success',
                    'endpoint': 'datasets',
                    'datasets_available': len(data) if isinstance(data, list) else 'multiple'
                }
                print(f"   ✅ SUCCESS: Elexon BMRS v1 API responding")
            else:
                # Try the demand data endpoint
                url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/INDOD"
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    results['elexon'] = {
                        'status': 'success',
                        'endpoint': 'INDOD',
                        'data_available': True
                    }
                    print("   ✅ SUCCESS: Elexon BMRS INDOD endpoint responding")
                else:
                    results['elexon'] = {'status': 'failed', 'error': f"HTTP {response.status_code}"}
                    print(f"   ❌ FAILED: HTTP {response.status_code}")
                
        except Exception as e:
            results['elexon'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ FAILED: {str(e)}")
        
        # Test 4: Carbon Intensity API (always free)
        print("\n4️⃣ Testing Carbon Intensity API...")
        try:
            url = f"{self.carbon_intensity_url}/intensity"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_intensity = data.get('data', [{}])[0].get('intensity', {})
                results['carbon_intensity'] = {
                    'status': 'success',
                    'current_forecast': current_intensity.get('forecast'),
                    'current_index': current_intensity.get('index')
                }
                print(f"   ✅ SUCCESS: Current carbon intensity - {current_intensity.get('forecast')} gCO2/kWh")
            else:
                results['carbon_intensity'] = {'status': 'failed', 'error': f"HTTP {response.status_code}"}
                print(f"   ❌ FAILED: HTTP {response.status_code}")
                
        except Exception as e:
            results['carbon_intensity'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ FAILED: {str(e)}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 API TEST SUMMARY")
        print("=" * 60)
        
        working_apis = [name for name, result in results.items() if result['status'] == 'success']
        print(f"✅ Working APIs: {len(working_apis)}/4")
        print(f"🔧 APIs available: {', '.join(working_apis)}")
        
        if len(working_apis) >= 3:
            print("🎉 Sufficient APIs working for energy prediction pipeline!")
        else:
            print("⚠️ Limited APIs available - may need troubleshooting")
        
        return results
    
    def ingest_neso_energy_data(self, start_date: str, end_date: str, max_requests: int = 50) -> Dict[str, Any]:
        """
        Ingest NESO energy data with request limits
        """
        try:
            self.logger.info(f"⚡ Ingesting NESO energy data: {start_date} to {end_date}")
            self.logger.info(f"📊 Max requests allowed: {max_requests}")
            
            # Convert dates for filtering
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            all_records = []
            requests_made = 0
            offset = 0
            limit = 1000  # Get more records per request
            
            while requests_made < max_requests:
                url = self.neso_base_url
                params = {
                    'resource_id': 'f6d02c0f-957b-48cb-82ee-09003f2ba759',
                    'limit': limit,
                    'offset': offset
                }
                
                response = requests.get(url, params=params, timeout=15)
                requests_made += 1
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('result', {}).get('records', [])
                    
                    if not records:
                        break
                    
                    # Filter records by date range
                    filtered_records = []
                    for record in records:
                        try:
                            record_date = datetime.strptime(record.get('SETTLEMENT_DATE', ''), '%Y-%m-%d')
                            if start_dt <= record_date <= end_dt:
                                filtered_records.append(record)
                        except:
                            continue
                    
                    all_records.extend(filtered_records)
                    offset += limit
                    
                    self.logger.info(f"📊 Request {requests_made}: {len(filtered_records)} relevant records")
                    
                    # Rate limiting - 1 request per second
                    time.sleep(1)
                    
                    # If we got fewer records than limit, we've reached the end
                    if len(records) < limit:
                        break
                else:
                    self.logger.error(f"❌ NESO API error: HTTP {response.status_code}")
                    break
            
            if all_records:
                df = pd.DataFrame(all_records)
                df = self._clean_neso_data(df)
                
                # Save to S3
                s3_key = self._generate_s3_key('neso_energy_data', start_date)
                self._save_to_s3(df, s3_key)
                
                return {
                    'status': 'success',
                    'records': len(df),
                    'requests_used': requests_made,
                    's3_location': f"s3://{self.config.raw_data_bucket}/{s3_key}",
                    'date_range': f"{df['settlement_date'].min()} to {df['settlement_date'].max()}"
                }
            else:
                return {
                    'status': 'failed', 
                    'error': 'No records found in date range',
                    'requests_used': requests_made
                }
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'requests_used': requests_made}
        """
        Ingest UK demand data from Elexon BMRS v1 API efficiently
        """
        try:
            self.logger.info(f"⚡ Ingesting Elexon demand data: {start_date} to {end_date}")
            self.logger.info(f"📊 Max requests allowed: {max_requests}")
            
            # Use INDOD (Indicative Operational Demand) dataset
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            all_records = []
            requests_made = 0
            
            # Get data day by day to control requests
            current_date = start_dt
            while current_date <= end_dt and requests_made < max_requests:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # INDOD endpoint for demand data
                url = f"{self.elexon_base_url}/datasets/INDOD"
                params = {
                    'settlementDate': date_str,
                    'format': 'json'
                }
                
                try:
                    response = requests.get(url, params=params, timeout=15)
                    requests_made += 1
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract records from response
                        if 'data' in data:
                            records = data['data']
                            all_records.extend(records)
                            self.logger.info(f"📊 Request {requests_made}: {len(records)} records for {date_str}")
                        else:
                            self.logger.warning(f"⚠️ No data structure found for {date_str}")
                    else:
                        self.logger.warning(f"⚠️ BMRS API error for {date_str}: HTTP {response.status_code}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Request failed for {date_str}: {str(e)}")
                
                current_date += timedelta(days=1)
                time.sleep(0.5)  # Small delay between requests
            
            if all_records:
                df = pd.DataFrame(all_records)
                df = self._clean_elexon_data(df)
                
                # Save to S3
                s3_key = self._generate_s3_key('elexon_demand_data', start_date)
                self._save_to_s3(df, s3_key)
                
                return {
                    'status': 'success',
                    'records': len(df),
                    'requests_used': requests_made,
                    's3_location': f"s3://{self.config.raw_data_bucket}/{s3_key}",
                    'date_range': f"{start_date} to {end_date}"
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'No demand data collected',
                    'requests_used': requests_made
                }
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'requests_used': requests_made}
    
    def _clean_elexon_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Elexon BMRS data"""
        # Convert dates and times
        if 'settlementDate' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['settlementDate'])
        
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        
        # Convert numeric columns
        numeric_cols = ['value', 'settlementPeriod']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename value to something more descriptive
        if 'value' in df.columns:
            df['demand_mw'] = df['value']
        
        return df.sort_values('timestamp' if 'timestamp' in df.columns else 'settlement_date')
        """
        Ingest NESO energy data with request limits
        """
        try:
            self.logger.info(f"⚡ Ingesting NESO energy data: {start_date} to {end_date}")
            self.logger.info(f"📊 Max requests allowed: {max_requests}")
            
            # Convert dates for filtering
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            all_records = []
            requests_made = 0
            offset = 0
            limit = 1000  # Get more records per request
            
            while requests_made < max_requests:
                url = self.neso_base_url
                params = {
                    'resource_id': 'f6d02c0f-957b-48cb-82ee-09003f2ba759',
                    'limit': limit,
                    'offset': offset
                }
                
                response = requests.get(url, params=params, timeout=15)
                requests_made += 1
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('result', {}).get('records', [])
                    
                    if not records:
                        break
                    
                    # Filter records by date range
                    filtered_records = []
                    for record in records:
                        try:
                            record_date = datetime.strptime(record.get('SETTLEMENT_DATE', ''), '%Y-%m-%d')
                            if start_dt <= record_date <= end_dt:
                                filtered_records.append(record)
                        except:
                            continue
                    
                    all_records.extend(filtered_records)
                    offset += limit
                    
                    self.logger.info(f"📊 Request {requests_made}: {len(filtered_records)} relevant records")
                    
                    # Rate limiting - 1 request per second
                    time.sleep(1)
                    
                    # If we got fewer records than limit, we've reached the end
                    if len(records) < limit:
                        break
                else:
                    self.logger.error(f"❌ NESO API error: HTTP {response.status_code}")
                    break
            
            if all_records:
                df = pd.DataFrame(all_records)
                df = self._clean_neso_data(df)
                
                # Save to S3
                s3_key = self._generate_s3_key('neso_energy_data', start_date)
                self._save_to_s3(df, s3_key)
                
                return {
                    'status': 'success',
                    'records': len(df),
                    'requests_used': requests_made,
                    's3_location': f"s3://{self.config.raw_data_bucket}/{s3_key}",
                    'date_range': f"{df['settlement_date'].min()} to {df['settlement_date'].max()}"
                }
            else:
                return {
                    'status': 'failed', 
                    'error': 'No records found in date range',
                    'requests_used': requests_made
                }
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'requests_used': requests_made}
    
    def ingest_weather_data_efficient(self, cities_limit: int = 8) -> Dict[str, Any]:
        """
        Ingest current weather for major UK cities efficiently
        """
        try:
            self.logger.info(f"🌤️ Ingesting weather data for {cities_limit} UK cities")
            
            uk_cities = [
                {'name': 'London', 'lat': 51.5074, 'lon': -0.1278},
                {'name': 'Birmingham', 'lat': 52.4862, 'lon': -1.8904},
                {'name': 'Manchester', 'lat': 53.4808, 'lon': -2.2426},
                {'name': 'Liverpool', 'lat': 53.4084, 'lon': -2.9916},
                {'name': 'Leeds', 'lat': 53.8008, 'lon': -1.5491},
                {'name': 'Sheffield', 'lat': 53.3811, 'lon': -1.4701},
                {'name': 'Bristol', 'lat': 51.4545, 'lon': -2.5879},
                {'name': 'Newcastle', 'lat': 54.9783, 'lon': -1.6178}
            ][:cities_limit]
            
            weather_data = []
            successful_requests = 0
            
            for city in uk_cities:
                try:
                    url = "http://api.openweathermap.org/data/2.5/weather"
                    params = {
                        'lat': city['lat'],
                        'lon': city['lon'],
                        'appid': self.weather_api_key,
                        'units': 'metric'
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        weather_record = {
                            'datetime': datetime.now(),
                            'city': city['name'],
                            'latitude': city['lat'],
                            'longitude': city['lon'],
                            'temperature_c': data['main']['temp'],
                            'humidity_percent': data['main']['humidity'],
                            'pressure_hpa': data['main']['pressure'],
                            'wind_speed_ms': data.get('wind', {}).get('speed', 0),
                            'wind_direction_deg': data.get('wind', {}).get('deg', 0),
                            'cloud_cover_percent': data.get('clouds', {}).get('all', 0),
                            'weather_description': data['weather'][0]['description'],
                            'visibility_m': data.get('visibility', 0)
                        }
                        weather_data.append(weather_record)
                        successful_requests += 1
                        
                    else:
                        self.logger.warning(f"⚠️ Weather API failed for {city['name']}: HTTP {response.status_code}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Weather request failed for {city['name']}: {str(e)}")
                
                time.sleep(0.2)  # Small delay to avoid rate limits
            
            if weather_data:
                df = pd.DataFrame(weather_data)
                
                # Save to S3
                s3_key = self._generate_s3_key('uk_weather_current', datetime.now().strftime('%Y-%m-%d'))
                self._save_to_s3(df, s3_key)
                
                return {
                    'status': 'success',
                    'records': len(df),
                    'successful_requests': successful_requests,
                    'total_requests': len(uk_cities),
                    's3_location': f"s3://{self.config.raw_data_bucket}/{s3_key}"
                }
            else:
                return {'status': 'failed', 'error': 'No weather data collected'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def ingest_carbon_intensity_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Ingest carbon intensity data (free API)
        """
        try:
            self.logger.info(f"🌱 Ingesting carbon intensity data: {start_date} to {end_date}")
            
            start_iso = f"{start_date}T00:00Z"
            end_iso = f"{end_date}T23:59Z"
            
            url = f"{self.carbon_intensity_url}/intensity/{start_iso}/{end_iso}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    carbon_records = []
                    for record in data['data']:
                        carbon_records.append({
                            'datetime_from': record['from'],
                            'datetime_to': record['to'],
                            'carbon_intensity_forecast': record['intensity']['forecast'],
                            'carbon_intensity_actual': record['intensity']['actual'],
                            'carbon_intensity_index': record['intensity']['index']
                        })
                    
                    df = pd.DataFrame(carbon_records)
                    df['datetime_from'] = pd.to_datetime(df['datetime_from'])
                    df['datetime_to'] = pd.to_datetime(df['datetime_to'])
                    
                    # Save to S3
                    s3_key = self._generate_s3_key('uk_carbon_intensity', start_date)
                    self._save_to_s3(df, s3_key)
                    
                    return {
                        'status': 'success',
                        'records': len(df),
                        's3_location': f"s3://{self.config.raw_data_bucket}/{s3_key}",
                        'date_range': f"{df['datetime_from'].min()} to {df['datetime_to'].max()}"
                    }
                else:
                    return {'status': 'failed', 'error': 'No carbon intensity data available'}
            else:
                return {'status': 'failed', 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _clean_neso_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NESO energy data"""
        # Convert dates
        if 'SETTLEMENT_DATE' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['SETTLEMENT_DATE'])
        
        # Convert numeric columns
        numeric_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND', 'SETTLEMENT_PERIOD']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create timestamp
        if 'settlement_date' in df.columns and 'SETTLEMENT_PERIOD' in df.columns:
            df['timestamp'] = df['settlement_date'] + pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='min')
        
        return df.sort_values('timestamp' if 'timestamp' in df.columns else 'settlement_date')
    
    def _generate_s3_key(self, data_type: str, date: str) -> str:
        """Generate S3 key with partitioning"""
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{data_type}/year={year}/month={month}/day={day}/{data_type}_{timestamp}.parquet"
    
    def _save_to_s3(self, df: pd.DataFrame, key: str) -> None:
        """Save DataFrame to S3"""
        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow')
        
        self.s3_client.put_object(
            Bucket=self.config.raw_data_bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        
        self.logger.info(f"💾 Saved {len(df)} records to s3://{self.config.raw_data_bucket}/{key}")

def run_efficient_pipeline_test():
    """
    Run the efficient pipeline with minimal API requests
    """
    print("🚀 EFFICIENT UK ENERGY PIPELINE")
    print("=" * 60)
    
    config = PipelineConfig()
    pipeline = EfficientUKEnergyPipeline(config)
    
    # Step 1: Test all APIs
    api_results = pipeline.test_all_apis()
    
    # Step 2: If APIs are working, do small data ingestion
    working_apis = [name for name, result in api_results.items() if result['status'] == 'success']
    
    if len(working_apis) >= 2:
        print("\n🚀 RUNNING SMALL DATA INGESTION TEST")
        print("-" * 60)
        
        # Test date range (small)
        start_date = "2024-06-01"
        end_date = "2024-06-02"
        
        ingestion_results = {}
        
        # NESO energy data (limited requests)
        if 'neso' in working_apis:
            print("⚡ Ingesting NESO energy data (max 10 requests)...")
            ingestion_results['neso'] = pipeline.ingest_neso_energy_data(start_date, end_date, max_requests=10)
        
        # Elexon BMRS demand data (if available)
        if 'elexon' in working_apis:
            print("🔌 Ingesting Elexon demand data (max 5 requests)...")
            ingestion_results['elexon'] = pipeline.ingest_elexon_demand_data(start_date, end_date, max_requests=5)
        
        # Weather data (8 cities max)
        if 'openweather' in working_apis:
            print("🌤️ Ingesting weather data (8 cities)...")
            ingestion_results['weather'] = pipeline.ingest_weather_data_efficient(cities_limit=8)
        
        # Carbon intensity (free)
        if 'carbon_intensity' in working_apis:
            print("🌱 Ingesting carbon intensity data...")
            ingestion_results['carbon'] = pipeline.ingest_carbon_intensity_data(start_date, end_date)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 INGESTION RESULTS")
        print("=" * 60)
        
        total_records = 0
        for source, result in ingestion_results.items():
            if result['status'] == 'success':
                records = result.get('records', 0)
                total_records += records
                print(f"✅ {source.upper()}: {records} records")
                if 'requests_used' in result:
                    print(f"   📊 API requests used: {result['requests_used']}")
            else:
                print(f"❌ {source.upper()}: {result['error']}")
        
        print(f"\n🎯 Total records ingested: {total_records}")
        print("💡 API usage optimized - ready for production!")
        
        return {'api_tests': api_results, 'ingestion': ingestion_results}
    else:
        print("\n⚠️ Insufficient working APIs for data ingestion")
        return {'api_tests': api_results}

if __name__ == "__main__":
    results = run_efficient_pipeline_test()