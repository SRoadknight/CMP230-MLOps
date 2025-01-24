# MLOps Pipeline -- CMP6230

This work was carried out for the module Data Management and Machine Learning Operations, as part of my undergraduate degree in Computer and Data Science. Changes have been made to reflect updates in the libraries.
<hr>
This repository provides instructions for setting up an MLOps pipeline using Docker containers, MariaDB ColumnStore, Redis, and a Conda environment with necessary Python packages for machine learning, data processing, and web API deployment.

## Prerequisites

- Docker
- Conda (for Python environment management)
- Python 3.8 or above
- Airflow
- MLFlow
- FastAPI
- Uvicorn

## Setup Instructions

Make sure the DAG is where Airflow is looking for DAGs. 

### 1. Run MariaDB ColumnStore inside Docker

Create a Docker container for MariaDB ColumnStore:

```bash
docker create --name mcs_container -p 3306:3306 mariadb/columnstore
```

### 2. Create Database User

Run the following command to create a user with privileges:

```bash
docker exec mcs_container mariadb -e "GRANT ALL PRIVILEGES ON *.* TO 'mlops'@'%' IDENTIFIED BY 'ChocolateRatings123';"
```

### 3. Create Database

Create the database for the pipeline:

```bash
CREATE DATABASE Chocolate_Bar_Ratings;
```

### 4. Run Redis Container

Create a Docker container for Redis:

```bash
docker create --name redis_store -p 6379:6379 redis
```

### 5. Setup Conda Environment

#### Update Conda

To ensure that Conda is up to date, run:

```bash
conda update -n base -c defaults conda
```

#### Create New Conda Environment

Create a new Conda environment with Python 3.8:

```bash
conda create -n mlops_pipeline python=3.8
```

Activate the environment:

```bash
conda activate mlops_pipeline
```

#### Install Required Python Packages

Install necessary Python packages from the Anaconda channel:

```bash
conda install -c anaconda pandas sqlalchemy pymysql scikit-learn redis
```

Install additional packages from Conda-Forge:

```bash
conda install -c conda-forge airflow mlflow fastapi uvicorn
```

### 6. Initialise and Run Airflow

For development purposes, initialise Airflow in standalone mode:

```bash
airflow standalone
```

### 7. Setup and Run MLFlow

```bash
mlflow server --backend-store-uri sqlite:///$HOME/mlflow/mlflow.db --default-artifact-root $HOME/mlflow/artifacts --host 0.0.0.0
```

### 8. Start Docker Containers

If you haven't already started the containers, run:

```bash
docker start mcs_container redis_store
```

### 9. Run FastAPI Application 

To run the FastAPI application locally, execute:

```bash
uvicorn app:main --reload
```
