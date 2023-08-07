Here I have provided all the datasets. 
You can split the original dataset into train and split using the train_flag column. With train_flag == 1 representing train data and train_flag == 0 representing test data.

Also some code read the file in parquet format. You can use xlsx format or you can easily convert the excel file to parquet using this code:


'''''''''''''''''''''''''''''''''''''''''
import pandas as pd

# Read the Excel file
df = pd.read_excel('/content/netbeans.xlsx')

# Convert the DataFrame to a Parquet file
df.to_parquet('/content/netbeans.parquet')

'''''''''''''''''''''''''''''''''''''''''''''''
