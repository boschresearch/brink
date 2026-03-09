import pandas as pd
import argparse
import os
from collections import defaultdict
import numpy as np


def parse_answer(x):
    try:
        parsed = eval(x)
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    return []


def multi_file_downsample(
        input_files,
        max_ratio=0.01,
        answer_column="answer",
        id_column="id",
        random_seed=42
):
    np.random.seed(random_seed)

    # Read all files and track source
    all_data = []
    file_mapping = {}

    print("Reading input files...")
    for i, file_path in enumerate(input_files):
        print(f"   Reading: {file_path}")
        df = pd.read_csv(file_path, sep="\t")
        df["parsed_answer"] = df[answer_column].apply(parse_answer)
        df = df[df["parsed_answer"].apply(lambda x: len(x) > 0)]

        for idx in df.index:
            file_mapping[len(all_data) + len(df.loc[:idx])] = i

        df["global_idx"] = range(len(all_data), len(all_data) + len(df))
        all_data.append(df)
        print(f"     Valid samples: {len(df)}")

    combined_df = pd.concat(all_data, ignore_index=True)
    total_samples = len(combined_df)

    print(f"\nGlobal statistics:")
    print(f"   Total files: {len(input_files)}")
    print(f"   Total samples: {total_samples}")

    # Calculate global frequency
    all_answers = combined_df["parsed_answer"].explode()
    freq = all_answers.value_counts()
    freq_ratio = freq / total_samples

    over_limit_entities = freq_ratio[freq_ratio > max_ratio].index.tolist()
    print(f"   Total entities: {len(freq)}")
    print(f"   Entities requiring downsampling: {len(over_limit_entities)}")

    # Global downsampling
    kept_indices = set()

    # Keep samples that don't need downsampling
    safe_mask = combined_df["parsed_answer"].apply(
        lambda ans: all(a not in over_limit_entities for a in ans)
    )
    safe_indices = set(combined_df[safe_mask].index)
    kept_indices.update(safe_indices)
    print(f"   Safe samples: {len(safe_indices)}")

    # Downsample entities that exceed the limit
    for entity in over_limit_entities:
        entity_mask = combined_df["parsed_answer"].apply(lambda ans: entity in ans)
        entity_rows = combined_df[entity_mask]

        allowed_count = int(max_ratio * total_samples)
        if len(entity_rows) > allowed_count:
            sampled_indices = np.random.choice(
                entity_rows.index,
                size=allowed_count,
                replace=False
            )
            kept_indices.update(sampled_indices)
            print(f"     Entity {entity}: {len(entity_rows)} -> {allowed_count}")
        else:
            kept_indices.update(entity_rows.index)
            print(f"     Entity {entity}: {len(entity_rows)} (no sampling needed)")

    # Update each file based on sampling results
    final_dfs = [[] for _ in input_files]
    kept_df = combined_df.loc[list(kept_indices)]

    print(f"\nAssigning sampling results to files:")
    for i, file_path in enumerate(input_files):
        original_df = pd.read_csv(file_path, sep="\t")
        original_df["parsed_answer"] = original_df[answer_column].apply(parse_answer)
        original_df = original_df[original_df["parsed_answer"].apply(lambda x: len(x) > 0)]

        file_kept = kept_df[kept_df.index.isin(all_data[i].index)]

        if len(file_kept) > 0:
            if id_column in original_df.columns:
                kept_ids = set(file_kept[id_column].values)
                final_df = original_df[original_df[id_column].isin(kept_ids)].copy()
            else:
                final_df = original_df.iloc[file_kept.index - all_data[i].index.min()].copy()

            if "parsed_answer" in final_df.columns:
                final_df.drop(columns=["parsed_answer"], inplace=True)
            if "global_idx" in final_df.columns:
                final_df.drop(columns=["global_idx"], inplace=True)

            final_dfs[i] = final_df
            print(f"   {os.path.basename(file_path)}: {len(original_df)} -> {len(final_df)}")
        else:
            print(f"   {os.path.basename(file_path)}: {len(original_df)} -> 0 (all filtered)")

    return final_dfs


def main():
    parser = argparse.ArgumentParser(
        description="Multi-file auto downsample with global fairness"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs='+',
        required=True,
        help="Paths to input TSV files (space-separated)"
    )
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=0.01,
        help="Max allowed ratio per answer entity globally (default 0.01)"
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="answer",
        help="Column name for answer (default 'answer')"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="id",
        help="Column name for unique ID (default 'id')"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_downsampled",
        help="Suffix for output files (default '_downsampled')"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    for file_path in args.input:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

    print("Starting multi-file joint downsampling...")
    print(f"   Max ratio: {args.max_ratio * 100}%")
    print(f"   Random seed: {args.random_seed}")

    result_dfs = multi_file_downsample(
        input_files=args.input,
        max_ratio=args.max_ratio,
        answer_column=args.answer_column,
        id_column=args.id_column,
        random_seed=args.random_seed
    )

    print(f"\nSaving downsampled results:")
    for i, (input_path, df) in enumerate(zip(args.input, result_dfs)):
        if len(df) > 0:
            try:
                df_sorted = df.sort_values(by=args.id_column, key=lambda x: pd.to_numeric(x, errors='coerce'))
                print(f"   Sorted by ID (numeric): {os.path.basename(input_path)}")
            except:
                df_sorted = df.sort_values(by=args.id_column)
                print(f"   Sorted by ID (string): {os.path.basename(input_path)}")

            base, ext = os.path.splitext(input_path)
            output_path = base + args.suffix + ext

            df_sorted.to_csv(output_path, sep="\t", index=False)
            print(f"   Saved: {output_path}")
        else:
            print(f"   Skipped empty file: {input_path}")

    print(f"\nMulti-file joint downsampling completed!")


if __name__ == "__main__":
    main()