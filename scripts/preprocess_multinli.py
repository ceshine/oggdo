import pandas as pd

if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    df = pd.read_csv("data/multinli/multinli_1.0_train.txt", sep="\t", error_bad_lines=False)
    # print(df["gold_label"].unique())
    print("Filtering out problematic rows...")
    print("Before:", df.shape[0])
    df = df[(df["gold_label"] != "-") & (~df["sentence1"].isnull()) & (~df["sentence2"].isnull())].copy()
    print("After:", df.shape[0])
    print("=" * 20)
    df["label"] = df["gold_label"]
    df["premise"] = df["sentence1"]
    df["hypo"] = df["sentence2"]
    df["id"] = df["pairID"]
    print("premise stats")
    print(df["premise"].str.len().describe())
    print(df.shape)
    df[["id", "hypo", "premise", "id", "label"]].to_csv('data/multinli/train.csv', index=False)
