import pandas as pd 
import os 

seasons =[  "0001", "0102", "0203", "0304", "0405", "0506", "0607", "0708", "0809", "0910",
    "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526","2627"
]

dfs = []

print("fetching data...")

for season in seasons:
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        df = pd.read_csv(url)
        df["Season_code"] = season
        dfs.append(df)
    except Exception as e:
        print(f"Error fetching data for season {season}: {e}")

raw_data = pd.concat(dfs, ignore_index=True)

os.makedirs("data/raw", exist_ok=True)
raw_data.to_csv("data/raw/raw_data.csv", index=False)