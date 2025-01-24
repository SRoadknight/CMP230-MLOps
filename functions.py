import numpy as np
import pandas as pd 
import pyarrow as pa
import pickle as pkl


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn


# Ingestion helper functions
# Extracts CSV from source file path
def extract_csv(path, logger):
	""" Extract CSV file if it exists. """

	try: 
		data = pd.read_csv(path)
		return data
	except Exception as e:
		logger.exception(
			f"Error: Unable to read the Flavors of Cacao dataset. {e}")

def write_df_to_db(conn, df, table_name):
	""" Write the Pandas dataframe to the selected table in the database. """

	df.to_sql(table_name, con=conn, if_exists="replace", index=False)

def rename_columns(df): 
	""" Rename the columns of the DataFrame. """

	new_cols = [
		'company_name', 
		'specific_bean_origin_or_bar_name', 
		'REF', 
		'review_year', 
		'cocoa_percent', 
		'company_loc',
		 'rating', 
		 'bean_type',
		  'broad_bean_origin'
	]
	df.columns = new_cols
	return df

# Ingestion function 
def ingestion(conn, logger, path, table_name): 
	""" Ingestion task for the Airflow DAG using the define helper functions. """

	def inner_func():

		df_extracted = extract_csv(path, logger)
		df_renamed = rename_columns(df_extracted)
		write_df_to_db(conn, df_renamed, table_name)
	return inner_func 


# Pre-processing helper functions

def read_df_from_db(db_conn, table_name):
	""" Read the data from the analytics database."""

	df = pd.read_sql(table_name, db_conn)
	return df

# Returns the features and target of the DataFrame as X and Y respectively
def X_Y_splitter(df, target_column):
	""" Splits the data into the feature column(s) as X and the target column(s) as Y. """

	X = df.drop(target_column, axis=1)
	Y = df[target_column]
	return X, Y

def df_in_range(df, bool_series):
	""" Creates a DataFrame that is a subset of the original based on the review_year column. """

	ranged_df = df[bool_series]
	return ranged_df
  
# Helper function to split dataframe into train, validate, and test sets
def train_validate_test_split(df, filter_column, target_column, train_date_max=2013, validate_date_max=2014):
	""" Splits the DataFrame into train, validate, and test based on the years to split on. """

	# Setup the conditions for each of the splits and get the boolean index
	upper_range_cond_train = df[filter_column] <= train_date_max
	range_cond_validate = (~upper_range_cond_train) & (df[filter_column] <= validate_date_max) 
	lower_range_cond_test = (~upper_range_cond_train) & (~range_cond_validate)

	# Using the boolean index's created to get the df's for train, validate, and test
	df_train = df_in_range(df, upper_range_cond_train)
	df_validate = df_in_range(df, range_cond_validate)
	df_test = df_in_range(df, lower_range_cond_test)

	# Split the train, validate, and test DataFrame into their X features and Y target.
	X_train, Y_train = X_Y_splitter(df_train, target_column)
	X_validate, Y_validate = X_Y_splitter(df_validate, target_column)
	X_test, Y_test = X_Y_splitter(df_test, target_column)
	return df_train, df_validate, df_test, X_train, X_validate, X_test, Y_train,\
	Y_validate, Y_test

def X_Y_concat(X, Y):
	""" Created a new DataFrame that includes the feature and target column(s) """

	X_y = pd.concat([X, Y], axis=1)
	return X_y

# Helper functions to pre-process the data 

def fill_missing(df):
	""" Replaces all no-break and NaN's with 'Unspecified'. """

	df = df.apply(lambda x: x.replace('\xa0', np.NaN))
	df = df.fillna("Unspecified")
	return df

def drop_columns(df, columns):
	""" Drop the specified columns from the DataFrame. """

	df = df.drop(columns, axis=1)
	return df

def encode_cocoa_percent(df):
	""" Encode cocoa_percent from a string to a float. """

	cocoa_percent = \
	(df['cocoa_percent'].apply(lambda x: str(x).replace('%', '')).astype(float)) / 100 
	return cocoa_percent

def blend_count(text):
	""" Helper function for parse_bean_type to determine to help determine if a bean_type
  is a blend or not"""

	bean_types = ["criollo", "trinitario", "nacional", "forastero"]
	if "blend" in text:
		return 0
	return sum(1 if bean_type in text else 0 for bean_type in bean_types)

def parse_bean_type(text):
	""" Parses the value and returns the variety of bean. """

	text = text.lower()
	count = blend_count(text)

	if any(x in text for x in ["blend", "amazon"]):
		return "Blend"
	if "trinitario" in text:
		return "Trinitario" if count == 1 else "Blend"
	if text.startswith("forastero"):
		return "Forastero"
	if text.startswith("nacional"):
		return "Nacional"
	if text.startswith("criollo"):
		return "Criollo"
	return "Unclassified"

# Preprocessing seen and unseen data
 
def preprocessing_seen(df): 
	"""  Pre-processing the seen data using the defined helper functions """

	df_preproc = fill_missing(df)
	df_preproc = drop_columns(df_preproc, ["REF", "specific_bean_origin_or_bar_name", "company_name", "company_loc", "broad_bean_origin"])
	df_preproc['cocoa_percent'] = encode_cocoa_percent(df_preproc)
	df_preproc['bean_type'] = df_preproc['bean_type'].apply(parse_bean_type)
	df_preproc = pd.get_dummies(df_preproc, drop_first=True, columns=['bean_type'])
	return df_preproc

def preprocessing_unseen(df_unseen, df_seen): 
	""" Pre-processing the unseen data, using the defined helper functions and some information discovered in the seen data. """

	# Using the created helper functions/leverage libraries to carry out the transformations
	df_preproc = fill_missing(df_unseen)
	df_preproc = drop_columns(df_preproc, ["REF", "specific_bean_origin_or_bar_name", "company_name", "company_loc", "broad_bean_origin"])
	df_preproc['cocoa_percent'] = encode_cocoa_percent(df_preproc)
	df_preproc['bean_type'] = df_preproc['bean_type'].apply(parse_bean_type)
	df_preproc = pd.get_dummies(df_preproc, drop_first=True, columns=['bean_type'])

	# Checks if any columns were seen in the training set but not the validation/testing set
	missing_cols = set(df_seen.columns) - set(df_preproc.columns)

	# Creates any missing columns setting all the values to 0 to show the absence 
	# (This would only heppen for our dummy encode columns)
	for c in missing_cols:
		df_preproc[c] = 0
	df_preproc = df_preproc[df_seen.columns]
	return df_preproc
    
# Main pre-processing function 
def preprocessing(db_conn, rdis_conn, table_name):
	""" Pre-processing function to be used in the Airflow DAG. """

	def inner_func():
		df_read = read_df_from_db(db_conn, table_name)

		# Using the created helper function to split the data into train, 
		# validate and test
		_, _, _, X_train, X_validate, X_test, Y_train, Y_validate, Y_test = \
		train_validate_test_split(df_read, "review_year", "rating")

		# Pre-processing steps here for both unseen and seen data
		X_train_preproc = preprocessing_seen(X_train)
		X_validate_preproc = preprocessing_unseen(X_validate, X_train_preproc)
		X_test_preproc = preprocessing_unseen(X_test, X_train_preproc)

		# Creating the df that contains both the pre-processed features and the target
		# If any columns records were dropped in the pre-processing the current function wouldn't work
		df_train_preproc = X_Y_concat(X_train_preproc, Y_train)
		df_validate_preproc = X_Y_concat(X_validate_preproc, Y_validate)
		df_test_preproc = X_Y_concat(X_test_preproc, Y_test)

		# Serialise df_train_preproc
		table = pa.Table.from_pandas(df_train_preproc)
		sink = pa.BufferOutputStream()
		with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
			writer.write_table(table)
		serialised_data = sink.getvalue().to_pybytes()
		rdis_conn.set("preproc_table_train", serialised_data)

		# Serialise df_validate_preproc
		table = pa.Table.from_pandas(df_validate_preproc)
		sink = pa.BufferOutputStream()
		with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
			writer.write_table(table)
		serialised_data = sink.getvalue().to_pybytes()
		rdis_conn.set("preproc_table_validate", serialised_data)

		# Serialise df_test_preproc
		table = pa.Table.from_pandas(df_test_preproc)
		sink = pa.BufferOutputStream()
		with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
			writer.write_table(table)
		serialised_data = sink.getvalue().to_pybytes()
		rdis_conn.set("preproc_table_validate", serialised_data)

		# We need some information about the preprocessing, for us it's only the expected columns
		# but in the future this could be any imputation means, scaling etc 
		# In this implementation we get the expected columns later on, but in a refined version I would do it here
		# and store it in Redis
	return inner_func

# Model training helper functions 

def eval_metrics(actual, pred):
	""" Calculate multiple evaluation metrics based on the actual target value and the predicted target value. """

	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)
	return rmse, mae, r2

# Model training main function
def training(rdis_conn, logger):
	""" Train the model using the pre-processed data, logging the training metrics to MLFlow and storing related Artfiacts. 
	Storing the created model, and run information in Redis for future retreival """

	def inner_func():
		# Get data from redis
		data_train = rdis_conn.get("preproc_table_train")
		buffer_train = pa.BufferReader(data_train)

		# Use open_stream to read the table from the buffer
		reader = pa.ipc.open_stream(buffer_train)
    
		# Read the table from the IPC stream
		table_train = reader.read_all()

		# Convert the Arrow Table to a pandas DataFrame
		train_df = table_train.to_pandas()

		# Get target and features
		X_train, Y_train = X_Y_splitter(train_df, "rating")
		# Initiating mlflow run
		run = mlflow.start_run()
		# run_id = str(run.info.run_id)
		# Create model (Using linear regression as a placeholder for now)
		mdl = LinearRegression() 
		
		# Fit the regression model
		mdl.fit(X_train, Y_train)     

		# Get the training results
		Y_train_pred = mdl.predict(X_train)  

		# Get metrics 
		rmse, mae, r2 = eval_metrics(Y_train, Y_train_pred)

		# Log information needed for pre-processing unseen data in the future
		# Right now this is from data available to us here, but this could also
		# be from Redis in the future
		mlflow.log_dict({"expected_columns": list(X_train.columns)}, "./model/preproc.json")
		logger.warning(mlflow.get_artifact_uri())
		
		
		# Log the metrics to mlflow
		mlflow.log_metric("train_rmse", rmse) # root mean square error
		mlflow.log_metric("train_r2", r2) # r squared
		mlflow.log_metric("train_mae", mae) # mean absolute error   

		
		# Serialise the data to redis
		# Let's use pickle to serialise the model as PyArrow is intended for Data
		serialised_mdl = pkl.dumps(mdl)
		serialised_run_id = pkl.dumps(run.info.run_id)
		
		# Store the model in redis
		rdis_conn.set("chocolate_bar_ratings_trained_mdl", serialised_mdl)
		
		# Store the MLFlow run id to continue the experiment run
		rdis_conn.set("chocolate_bar_ratings_trained_run_id", serialised_run_id)
		
	return inner_func

# Model evaluation main function
def evaluate(rdis_conn):
	""" Evaluate the model, generating the evaluation metrics and logging them to the run in MLFlow"""

	def inner_func():
		# Get data from redis
		# Getting the model
		mdl = pkl.loads(rdis_conn.get("chocolate_bar_ratings_trained_mdl"))
		
		# Getting the run id for MLFlow to continue the experiment
		retrieved_run_id = pkl.loads(rdis_conn.get("chocolate_bar_ratings_trained_run_id"))
		
		# Getting the validate data 
		data_validate = rdis_conn.get("preproc_table_train")
		buffer_validate = pa.BufferReader(data_validate)
		reader = pa.ipc.open_stream(buffer_validate)
		table_validate = reader.read_all()
		train_df = table_validate.to_pandas()
		
		
		X_validate, Y_validate = X_Y_splitter(train_df, "rating")
		# X_test, Y_test = X_Y_splitter(data_test, "rating")
		
		# Resume the previously started experiment run
		run = mlflow.start_run(run_id=retrieved_run_id)
		
		# Perform the prediction
		Y_validate_predict = mdl.predict(X_validate)
		# Y_test_predict = mdl.predict(X_test)
		
		# Evaluate the results     
		validate_rmse, validate_mae, validate_r2 = eval_metrics(Y_validate, Y_validate_predict)
		# test_rmse, test_mae, test_r2 = eval_metrics(Y_test, Y_test_predict)
		
		# Log the metrics to mlflow
		mlflow.log_metric("validate_rmse", validate_rmse) # root mean square error
		mlflow.log_metric("validate_r2", validate_r2) # r squared
		mlflow.log_metric("validate_mae", validate_mae) # mean absolute error  
		tracking_url_type_filestore = urlparse(mlflow.get_tracking_uri()).scheme

		# model logging
		# Model registry does not work with file store
		if tracking_url_type_filestore != "file":            
			# Registering the model
			# Please refer to the documentation for further information:
			# https://mlflow.org/docs/la/model-registry.html#api-workflow
			mlflow.sklearn.log_model(mdl, "model", registered_model_name="ChocolateBarRating_LinearRegrModel")
		else:
			mlflow.sklearn.log_model(mdl, "model")
			
			# End the current experiment run
		mlflow.end_run()    
	return inner_func