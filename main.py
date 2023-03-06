# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:27:07 2023

@author: om
"""
#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
# 2. Create the app object
app = FastAPI()

from transformers import T5Tokenizer, T5ForConditionalGeneration

loaded_tokenizer = T5Tokenizer.from_pretrained('/tokenizer_dir/tokenizer.pkl')
# loaded_tokenizer = tokenizer.save_pretrained('/tokenizer_dir/tokenizer.pkl')

# loaded_tokenizer = T5Tokenizer.from_pretrained('/tokenizer_dir/tokenizer.pkl')

# loaded_model = T5ForConditionalGeneration.from_pretrained('t5-large')
loaded_model = pickle.load(open("model_large.pkl", "rb"))

class request_body(BaseModel):
    text:str
    num_sentences: str

@app.post('/api/v1/summerize')
async def get_summary(data: request_body, num_sentences=40):
    inputs = loaded_tokenizer.encode("summarize: " + data.text, return_tensors="pt", truncation=True)
    outputs = loaded_model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)#change max_length to get the desired number of sentence as output
    summary = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_sentences = int(num_sentences)
    summary = ' '.join(summary.split()[:num_sentences])        
    return {
        "response": summary
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn <filename>:app --reload
