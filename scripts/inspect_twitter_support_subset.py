import pandas as pd

def main():
    df = pd.read_parquet("data/raw/twitter_support_subset.parquet")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nSample rows:\n", df.head(3).to_dict(orient="records"))

if __name__ == "__main__":
    main()
