import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from main import ask_question  # Your RAG function

# 1. Define scope and credentials
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

# 2. Open the Google Sheet
sheet = client.open("MyDatabaseSheet").sheet1  

# 3. Load data into a DataFrame
data = sheet.get_all_records()
df = pd.DataFrame(data)

print("Current Google Sheet Data:")
print(df)

# 4. Loop through rows and fill missing answers
for idx, row in df.iterrows():
    question = row.get("Question", "").strip()
    answer = row.get("Answer", "").strip()

    if question and not answer:
        # Call your RAG function
        rag_answer = ask_question(question)

        # Update DataFrame
        df.at[idx, "Answer"] = rag_answer

        # Update Google Sheet (row index +2 because sheet rows start at 1 and header is row 1)
        sheet.update_cell(idx + 2, df.columns.get_loc("Answer") + 1, rag_answer)

        print(f"Answered: {question} -> {rag_answer}")

print("\nAll unanswered questions have been processed and updated in Google Sheets!")
