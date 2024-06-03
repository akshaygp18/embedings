import pandas as pd
import json
import tiktoken
import os

from utils import get_embedding

# Configuration
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

# # Load and process JSON data
# def process_json(json_path="entries.json", top_n=1000):
#     with open(json_path, 'r') as file:
#         json_data = json.load(file)

#     # Assuming JSON data is a list of dictionaries
#     df_json = pd.DataFrame(json_data)
#     df_json['combined'] = df_json.apply(lambda row: json.dumps(row), axis=1)

#     encoding = tiktoken.get_encoding(embedding_encoding)
#     df_json["n_tokens"] = df_json.combined.apply(lambda x: len(encoding.encode(x)))
#     df_json = df_json[df_json.n_tokens <= max_tokens].tail(top_n)

#     # Generate embeddings
#     df_json["embedding"] = df_json.combined.apply(lambda x: get_embedding(x, model=embedding_model))
    
#     # Save to CSV
#     df_json.to_csv("json_data_with_embeddings.csv", index=False)
#     return df_json

# # Load and process Excel data
# def process_excel(csv_path, top_n=1000):
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"The file {csv_path} does not exist.")
#     df_excel = pd.read_csv(csv_path)
    
#     # Assuming specific columns for processing based on the image provided
#     df_excel['combined'] = df_excel.apply(lambda row: f"{row[1]} {row[2]}: {row[3]}", axis=1)

#     encoding = tiktoken.get_encoding(embedding_encoding)
#     df_excel["n_tokens"] = df_excel.combined.apply(lambda x: len(encoding.encode(x)))
#     df_excel = df_excel[df_excel.n_tokens <= max_tokens].tail(top_n)

#     # Generate embeddings
#     df_excel["embedding"] = df_excel.combined.apply(lambda x: get_embedding(x, model=embedding_model))
    
#     # Save to CSV
#     df_excel.to_csv("excel_data_with_embeddings.csv", index=False)
#     return df_excel

# # Example usage
# json_path = "data.json"  # Replace with your JSON file path
# csv_path = "d1.csv"  # Replace with your Excel file path

# # df_json = process_json(json_path)
# df_excel = process_excel(csv_path)

# # print("JSON Data with Embeddings:", df_json.head())
# print("Excel Data with Embeddings:", df_excel.head())

import numpy as np
import json
from sentence_transformers import SentenceTransformer

def load_text_embeddings(text_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_data)
    return embeddings

def main():
    # Load new JSON data
    with open('entries .json', 'r') as file:
        new_json_data = json.load(file)
    
    # Extract text data for embeddings
    new_text_data = [json.dumps(item) for item in new_json_data]
    new_text_embeddings = load_text_embeddings(new_text_data)
    
    # Save new embeddings
    np.save('new_text_embeddings.npy', new_text_embeddings)

if __name__ == "__main__":
    main()

