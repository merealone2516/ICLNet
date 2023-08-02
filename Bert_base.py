# import necessary libraries
import pandas as pd 
import re
import torch
import torchvision
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
from datetime import datetime

# set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set use of maxpool
use_maxpool = False

# list of files to be processed. Here you can add the dataset files like original train or test file and modified train and test file, to get the embeddings.
files = [
    "/home/atif/abhi/Netbeans/netbeans_original_dataset_training_date.xlsx",
    "/home/atif/abhi/Cassandra/cassandra_original_dataset_training_date.xlsx",
    "/home/atif/abhi/Freemarker/freemarker_original_dataset_training_date.xlsx"        
]

# process each file in the list
for file in files:
    # read the Excel file into a dataframe
    df = pd.read_excel(file)
    print("\n\n", file, "\n\n")
    
    # format date columns
    df['author_time_date'] = df['author_time_date'].dt.strftime("%m/%d/%Y")
    df['commit_time_date'] = df['commit_time_date'].dt.strftime("%m/%d/%Y")
    df['created_date'] = df['created_date'].dt.strftime("%m/%d/%Y")
    df['updated_date'] = df['updated_date'].dt.strftime("%m/%d/%Y")
    
    # convert columns to lists
    test_data0 = df['summary_processed'].tolist()
    # ... truncated for brevity

    # load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True,).to(device)
    batch_size = 256

    # function to get BERT embeddings
    def get_bert_embeddings(data, model, tokenizer, batch_size):
        # loop over data in chunks of batch_size
        for i in tqdm(range(0, len(data), batch_size)):
            # tokenize the batch of data
            bert_input = tokenizer(data[i:i+batch_size], padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
            
            # get model outputs
            with torch.no_grad():
                last_hidden_states = model(bert_input['input_ids'].to(device), bert_input['token_type_ids'].to(device))
                # get max pool across time dimension
                max_pool, _ = torch.max(last_hidden_states[0], dim=2)
            
            # concat embeddings and max pool, or use only embeddings
            if use_maxpool:
                feats = torch.concat((last_hidden_states[0][:,0,:], max_pool), axis=1).cpu().detach().numpy()
            else:
                feats = last_hidden_states[0][:,0,:].cpu().detach().numpy()
            
            # append features to total
            if i == 0:
                total_features = feats
            else:
                total_features = np.concatenate((total_features, feats), axis=0)
        
        return total_features

    # get BERT embeddings for each data list
    features0 = get_bert_embeddings(test_data0, model, tokenizer, batch_size)
    features1 = get_bert_embeddings(test_data1, model, tokenizer, batch_size)
    features2 = get_bert_embeddings(test_data2, model, tokenizer, batch_size)
    features3 = get_bert_embeddings(test_data3, model, tokenizer, batch_size)
    features4 = get_bert_embeddings(test_data4, model, tokenizer, batch_size)
    features7 = get_bert_embeddings(test_data7, model, tokenizer, batch_size)
    features8 = get_bert_embeddings(test_data8, model, tokenizer, batch_size)
    features11 = get_bert_embeddings(test_data11, model, tokenizer, batch_size)
    features12 = get_bert_embeddings(test_data12, model, tokenizer, batch_size)
    

    
    # concatenate all features
    final_features = np.concatenate((features0, features1, features2,features3, features4, features7, features8, features11, features12),axis=1)
    print(final_features.shape)

    # save features to file
    if use_maxpool:
        np.savetxt(file.split(".")[0]+"_emb_maxpool_non-textual3.csv", final_features, delimiter=",")
    else:
        np.savetxt(file.split(".")[0]+"_emb_textual.csv", final_features, delimiter=",")
