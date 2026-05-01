import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import argparse


def preprocess_and_sample(sample_fraction=0.99, random_state=42):
    print("Đang tải BANKING77 dataset...")
    dataset = load_dataset("PolyAI/banking77")

    df_train_full = dataset["train"].to_pandas()
    df_test_full = dataset["test"].to_pandas()

    features = dataset["train"].features
    label_names = features["label"].names

    # Map label ID sang text intent để dễ theo dõi
    df_train_full["intent"] = df_train_full["label"].apply(lambda x: label_names[x])
    df_test_full["intent"] = df_test_full["label"].apply(lambda x: label_names[x])

    print(f"Đang sampling {sample_fraction*100}% dữ liệu training...")
    df_train_sampled, _ = train_test_split(
        df_train_full,
        train_size=sample_fraction,
        stratify=df_train_full["label"],
        random_state=random_state,
    )

    # Tiền xử lý: Dọn dẹp khoảng trắng thừa
    df_train_sampled["text"] = df_train_sampled["text"].str.strip()
    df_test_full["text"] = df_test_full["text"].str.strip()

    os.makedirs("sample_data", exist_ok=True)

    train_path = "sample_data/train.csv"
    test_path = "sample_data/test.csv"

    df_train_sampled[["text", "label", "intent"]].to_csv(train_path, index=False)
    df_test_full[["text", "label", "intent"]].to_csv(test_path, index=False)

    print(f"Đã lưu tập train ({len(df_train_sampled)} mẫu) vào {train_path}")
    print(f"Đã lưu tập test ({len(df_test_full)} mẫu) vào {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fraction", type=float, default=0.99, help="Tỷ lệ dữ liệu training cần sample"
    )
    args = parser.parse_args()
    preprocess_and_sample(sample_fraction=args.fraction)
