#!/usr/bin/env python3
"""Test MLflow connection with environment variables"""

import os
import sys
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_mlflow_connection():
    """Test MLflow connection using environment variables"""
    
    # Load environment variables
    load_dotenv(".env")
    
    print("Testing MLflow connection...")
    print("=" * 50)
    
    # Get environment variables
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    gcp_project = os.getenv("GCP_PROJECT_ID")
    
    print(f"Tracking URI: {tracking_uri}")
    print(f"Artifact URI: {artifact_uri}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password) if password else 'None'}")
    print(f"GCP Project: {gcp_project}")
    print()
    
    if not tracking_uri:
        print("❌ MLFLOW_TRACKING_URI not set in environment")
        return False
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow tracking URI set: {tracking_uri}")
        
        # Create client
        client = MlflowClient()
        print("✅ MLflow client created successfully")
        
        # Test connection by listing experiments
        experiments = client.search_experiments()
        print(f"✅ Successfully connected! Found {len(experiments)} experiments:")
        
        for exp in experiments[:5]:  # Show first 5 experiments
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Test creating/getting experiment
        experiment_name = "thomas/DiffusionDet_Connection_Test"
        
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=artifact_uri,
                    tags={"test": "connection", "framework": "accelerate"}
                )
                print(f"✅ Created test experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✅ Found existing experiment: {experiment_name}")
            
            # Test creating a run
            run = client.create_run(
                experiment_id=experiment_id,
                run_name="connection_test",
                tags={
                    "test": "true",
                    "framework": "accelerate",
                    "project": gcp_project
                }
            )
            
            print(f"✅ Created test run: {run.info.run_id}")
            
            # Test logging parameters and metrics
            client.log_param(run.info.run_id, "test_param", "test_value")
            client.log_metric(run.info.run_id, "test_metric", 0.123, step=1)
            
            print("✅ Successfully logged test parameter and metric")
            
            # End the test run
            client.set_terminated(run.info.run_id, "FINISHED")
            print("✅ Test run completed successfully")
            
            print()
            print("🎉 MLflow connection test PASSED!")
            print(f"You can view the test run at: {tracking_uri}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create experiment or run: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to MLflow: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test required imports"""
    print("Testing imports...")
    print("-" * 30)
    
    try:
        import mlflow
        print("✅ mlflow imported successfully")
        
        from mlflow import MlflowClient
        print("✅ MlflowClient imported successfully")
        
        from dotenv import load_dotenv
        print("✅ dotenv imported successfully")
        
        import urllib3
        print("✅ urllib3 imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("MLflow Connection Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed!")
        sys.exit(1)
    
    print()
    
    # Test MLflow connection
    if test_mlflow_connection():
        print()
        print("✅ All tests passed! MLflow is properly configured.")
        sys.exit(0)
    else:
        print()
        print("❌ MLflow connection test failed!")
        sys.exit(1)
