import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]


creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)


sheet = client.open("MyDatabaseSheet").sheet1  

data = sheet.get_all_records()
df = pd.DataFrame(data)

print(" Current Google Sheet Data:")
print(df)

for idx, row in df.iterrows():
    question = row["Question"]
    answer = row.get("Answer", "")

    
    if not answer:
     
        from main import ask_question  
        rag_answer = ask_question(question)

      
        df.at[idx, "Answer"] = rag_answer

       
        sheet.update_cell(idx+2, 2, rag_answer)  

        print(f" Answered: {question} -> {rag_answer}")

print("\n All unanswered questions have been processed and updated in Google Sheets!")
