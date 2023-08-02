# Import necessary libraries
import pandas as pd
import re
import torch
import torchvision
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a flag for using max pooling or not
use_maxpool = False

# File paths for the datasets
files = ["/home/atif/abhi/Cassandra/cass_pad.xlsx",
         "/home/atif/abhi/Arrow/Arrow_dataset_pad.xlsx",
         ]
# Process each file
for file in files:
    df = pd.read_excel(file)
    print("\n\n", file, "\n\n")
  
    # Collect necessary columns from the dataframes
    test_data0 = df['summary_processed'].tolist()
    test_data1 = df['description_processed'].tolist()
    test_data2 = df['message_processed'].tolist()
    test_data11 = df['changed_files'].tolist()
    test_data12 = df['processDiffCode'].tolist()
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    model = AutoModel.from_pretrained('albert-base-v2', output_hidden_states=True).to(device)
    batch_size = 256
    # Function to get BERT embeddings
    def get_bert_embeddings(data, model, tokenizer, batch_size):
        for i in tqdm(range(0, len(data), batch_size)):
            bert_input = tokenizer(data[i:i+batch_size], padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            with torch.no_grad():
                last_hidden_states = model(bert_input['input_ids'].to(device))
                max_pool, _ = torch.max(last_hidden_states[0], dim=1)
            if use_maxpool:
                feats = torch.cat((last_hidden_states[0][:,0,:], max_pool), axis=1).cpu().detach().numpy()
            else:
                feats = last_hidden_states[0][:,0,:].cpu().detach().numpy()
            if i == 0:
                total_features = feats
            else:
                total_features = np.concatenate((total_features, feats), axis=0)
        return total_features

    # Get embeddings for each set of data
    features0 = get_bert_embeddings(test_data0, model, tokenizer, batch_size)
    features1 = get_bert_embeddings(test_data1, model, tokenizer, batch_size)
    features2 = get_bert_embeddings(test_data2, model, tokenizer, batch_size)
    features11 = get_bert_embeddings(test_data11, model, tokenizer, batch_size)
    features12 = get_bert_embeddings(test_data12, model, tokenizer, batch_size)

    # Concatenate all features
    final_features = np.concatenate((features0, features1, features2, features11, features12),axis=1)
    print(final_features.shape)
    if use_maxpool:
        np.savetxt(file.split(".")[0]+"_emb_maxpool_non-textual3.csv", final_features, delimiter=",")
    else:
        np.savetxt(file.split(".")[0]+"_albert_emb_textual.csv", final_features, delimiter=",")
