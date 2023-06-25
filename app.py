#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import uvicorn
model = joblib.load('hyundai_price_predict_pipe.pkl')


# In[2]:


#define inputclass
app = FastAPI()
class my_input(BaseModel):
    model: object
    year: int
    transmission: object
    mileage: int
    fuelType: object
    tax:int
    mpg:float
    engineSize:float


# In[3]:


# define request body

@app.post('/predict/')
async def main(input: my_input):
    
    data = input.dict()
    data_ = [[data['model'], data['year'], data['transmission'], data['mileage'], data['fuelType'],data['tax'],
         data['mpg'],data['engineSize']]]

    prediction = model.predict(data_)[0]


    return {
    'prediction': prediction
    }


# In[ ]:




