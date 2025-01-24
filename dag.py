from datetime import datetime, timedelta 
import sys
import os

# Airflow imports
from airflow import DAG 
from airflow.operators.python import PythonOperator 
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))
from functions import ingestion, preprocessing, training, evaluate 

# Other imports 
import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine 
import logging
import mlflow

import redis

# Globals 
DEFAULT_INGEST_PATH = "~/airflow/dags/chocolate-bar-ratings-ml-pipeline/data/flavors_of_cacao.csv" 
DEFAULT_TABLE_NAME="Chocolate_Bar_Ratings"

# Required connections and serialisation context created 
mcs_conn = create_engine("mysql+pymysql://mlops:ChocolateRatings123@localhost:3306/Chocolate_Bar_Ratings")
redis_conn = redis.Redis(host="127.0.0.1", port=6379)
mlflow.set_tracking_uri("http://localhost:5000")


# Set up logging of important data 
logging.basicConfig(level=logging.WARN) 
logger = logging.getLogger(__name__)



default_args = { 
    "owner": "Chocolate Ratings", 
    "depends_on_past": False,
    "email": ["chocolate@ratings.com"], 
    "email_on_failure": False, 
    "email_on_retry": False, 
    "retries": 0,
    "retry_delay": timedelta(minutes=5) 
}

with DAG(
    "ETL", 
    default_args=default_args,
    description="Data ingestion", 
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10), 
    catchup=False, 
    tags=["ingestion", "preporcessing", "training", "evaluation"], 
) as dag: 
    ingestion_task = PythonOperator(
        task_id="ingestion", 
        python_callable=ingestion(mcs_conn, logger, DEFAULT_INGEST_PATH , DEFAULT_TABLE_NAME) 
    )
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocessing(mcs_conn, redis_conn, DEFAULT_TABLE_NAME) 
    )
    model_training_task = PythonOperator(
        task_id="model_training", 
        python_callable=training(redis_conn, logger) 
    ) 
    model_evaluation_task = PythonOperator(
        task_id="model_evaluation", 
        python_callable=evaluate(redis_conn) 
    )



ingestion_task >> preprocessing_task >> model_training_task >> model_evaluation_task