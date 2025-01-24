from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow.pyfunc

app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")
model_name = "ChocolateBarRating_LinearRegrModel"
model_version = 1
    
mdl = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


# Helper functions to pre-process the data
def fill_missing(df):
	"""
	"""
	df = df.apply(lambda x: x.replace('\xa0', np.NaN))
	df = df.fillna("Unspecified")
	return df

def drop_columns(df, columns):
	"""
	"""
	df = df.drop(columns, axis=1)
	return df

def encode_cocoa_percent(df):
	"""
	"""
	cocoa_percent = \
	(df['cocoa_percent'].apply(lambda x: str(x).replace('%', '')).astype(float)) / 100 
	return cocoa_percent

def blend_count(text):
	"""
	"""
	bean_types = ["criollo", "trinitario", "nacional", "forastero"]
	if "blend" in text:
		return 0
	return sum(1 if bean_type in text else 0 for bean_type in bean_types)

def parse_bean_variety(text):
	"""
	"""
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

def preprocessing(df): 
	expected_columns = ['review_year', 'cocoa_percent', 'bean_type_Criollo',
       'bean_type_Forastero', 'bean_type_Nacional', 'bean_type_Trinitario',
       'bean_type_Unclassified']
	df_preproc = fill_missing(df)
	df_preproc = drop_columns(df_preproc, ["REF", "specific_bean_origin_or_bar_name", "company_name", "company_loc", "broad_bean_origin"])
	df_preproc['cocoa_percent'] = encode_cocoa_percent(df_preproc)
	df_preproc['bean_type'] = df_preproc['bean_type'].apply(parse_bean_variety)
	df_preproc = pd.get_dummies(df_preproc, drop_first=True, columns=['bean_type'])
	missing_cols = set(expected_columns) - set(df_preproc.columns)
	for c in missing_cols:
		df_preproc[c] = 0
	df_preproc = df_preproc[expected_columns]
	return df_preproc

class ChocolateInput(BaseModel):
    company_name: str 
    specific_bean_origin_or_bar_name: str 
    REF: int 
    review_year: int 
    cocoa_percent: str 
    company_loc: str 
    bean_type: str 
    broad_bean_origin: str 

class ChocolatePrediction(BaseModel):
    company_name: str 
    specific_bean_origin_or_bar_name: str 
    REF: int 
    review_year: int 
    cocoa_percent: str 
    company_loc: str 
    bean_type: str 
    broad_bean_origin: str
    rating: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chocolate/predict/single")
def predict_chocolate_single(data: ChocolateInput):
    data_dict = data.model_dump()

    company_name = data_dict['company_name'] 
    specific_bean_origin_or_bar_name = data_dict['specific_bean_origin_or_bar_name'] 
    REF = data_dict['REF'] 
    review_year = data_dict['review_year'] 
    cocoa_percent = data_dict['cocoa_percent'] 
    company_loc = data_dict['company_loc'] 
    bean_type = data_dict['bean_type'] 
    broad_bean_origin = data_dict['broad_bean_origin'] 
    
    df = pd.DataFrame.from_dict([data_dict])
    df_preproc = preprocessing(df)
    prediction = mdl.predict(df_preproc)
    data_dict['rating'] = round(prediction[0], 2)
    chocolate_prediction = ChocolatePrediction.model_validate(data_dict)
    return {"Chocolate Prediction": chocolate_prediction}

        
        
        
        
        
        
        
        
        
        