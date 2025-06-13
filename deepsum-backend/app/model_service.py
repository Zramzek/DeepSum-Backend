import os
import json

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, T5Tokenizer
from typing import List

from config import MODEL_PATH


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_input_length = 512
        self.max_output_length = 64
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> tf.Tensor:
        input_text = "summarize: " + text
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='tf', 
            max_length=self.max_input_length, 
            truncation=True
        )
        return input_ids

    def generate_summary(self, text: str) -> str:
        chunks = [text[i:i+self.max_input_length] for i in range(0, len(text), self.max_input_length)]
        
        summaries = []
        for chunk in chunks:
            input_ids = self.preprocess_text(chunk)
            summary_ids = self.model.generate(
                input_ids, 
                max_length=self.max_output_length
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        final_summary = " ".join(summaries)
        return final_summary


model_service = ModelService()