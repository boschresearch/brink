#!/usr/bin/env python3
import pandas as pd
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split first n rows of TSV file into train/val/test with 8:1:1 ratio"
    )
    parser.add_argument("--input", type=str, required=True, help="Input TSV file path")
    parser.add_argument("--output", type=str, default="splits", help="Output directory (default: splits)")
    parser.add_argument("--nrows", type=int, default=300, help="Number of rows to take from input (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        data = pd.read_csv(args.input, sep='\t')
    except Exception as e:
        print(f"Error reading file {args.input}: {e}")
        exit(1)

    if len(data) < args.nrows:
        print(f"Error: Input file has only {len(data)} rows, but {args.nrows} requested.")
        exit(1)

    # Take first nrows
    data = data.head(args.nrows)

    # Shuffle data
    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split according to 8:1:1 ratio
    total = len(data)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    os.makedirs(args.output, exist_ok=True)

    train_path = os.path.join(args.output, "train.tsv")
    val_path = os.path.join(args.output, "val.tsv")
    test_path = os.path.join(args.output, "test.tsv")

    train_data.to_csv(train_path, sep='\t', index=False)
    val_data.to_csv(val_path, sep='\t', index=False)
    test_data.to_csv(test_path, sep='\t', index=False)

    print(f"Dataset split successfully: train {len(train_data)}, val {len(val_data)}, test {len(test_data)}")
    print(f"Files saved in directory: {args.output}")

if __name__ == "__main__":
    main()