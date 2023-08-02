df = pd.read_excel('Enter the file here')

df = df.fillna('PAD')

spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","\n","↓","↑","$"]

for char in spec_chars:
    df['summary_processed'] = df['summary_processed'].str.replace(char, ' ')
    df['description_processed'] = df['description_processed'].str.replace(char, ' ')
    df['message_processed'] = df['message_processed'].str.replace(char, ' ')
    df['changed_files'] = df['changed_files'].str.replace(char, ' ')
    df['processDiffCode'] = df['processDiffCode'].str.replace(char, ' ')

# Convert the date strings to datetime objects
df['created_date'] = pd.to_datetime(df['created_date'], format="%Y-%m-%d %H:%M:%S")
df['updated_date'] = pd.to_datetime(df['updated_date'], format="%Y-%m-%d %H:%M:%S")
df['author_time_date'] = pd.to_datetime(df['author_time_date'], format="%Y-%m-%d %H:%M:%S")
df['commit_time_date'] = pd.to_datetime(df['commit_time_date'], format="%Y-%m-%d %H:%M:%S")

#Save
df.to_excel("write file name you want to save with")
