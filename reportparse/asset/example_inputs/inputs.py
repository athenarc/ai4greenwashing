import os
import pandas as pd
import requests

REPORTS_SAVE_PATH = "reportparse/asset/example_input"

df = pd.read_json("data/reports.json")


# function to download files
def download_files(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for url in df["pdf_url"]:
        pdf_filename = os.path.basename(url)
        response = requests.get(url)
        with open(os.path.join(save_dir, pdf_filename), "wb") as file:
            file.write(response.content)
    print(f"Success.")


# download_files(df, REPORTS_SAVE_PATH)
# download_files(df, REPORTS_SAVE_PATH)

# download subset of reports
df_sample = df[df["company_name"] == "Apple"]
download_files(df_sample, REPORTS_SAVE_PATH)
