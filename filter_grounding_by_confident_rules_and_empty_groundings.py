#!/usr/bin/env python3

import os
import json
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Filter groundings files based on rule_index")
    parser.add_argument("--filtered_rule_dir", required=True,
                        help="Directory with filtered rule JSON files")
    parser.add_argument("--original_grounding_dir", required=True,
                        help="Original groundings directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for filtered groundings")
    return parser.parse_args()

def load_rule_indexes(json_file_path):
    """Extract rule_index from filtered rule JSON file."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    indexes = set()
    for item in data:
        if "rule_index" in item:
            indexes.add(str(item["rule_index"]))
    return indexes

def parse_rule_index_from_filename(filename):
    """Parse rule index from filename like 'rule_12_groundings.tsv'."""
    base, ext = os.path.splitext(filename)
    if not base.startswith("rule_") or not base.endswith("_groundings"):
        return None
    middle = base[len("rule_"):]
    if middle.endswith("_groundings"):
        middle = middle[:-len("_groundings")]
    else:
        return None
    return middle

def main():
    args = parse_args()

    filtered_rule_dir = args.filtered_rule_dir
    original_grounding_dir = args.original_grounding_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    rule_types = [
        "composition_rules",
        "inversion_rules",
        "intersection_rules",
        "hierarchy_rules",
        "diamond_rules",
        "triangle_rules",
        "symmetry_rules",
        "other_rules",
        "3hop_rules",
        "4hop_rules"
    ]

    for rule_type in rule_types:
        base_json = os.path.join(filtered_rule_dir, f"{rule_type}.json")
        alt_json = os.path.join(filtered_rule_dir, f"{rule_type.replace('_rules', '_rules_with_index')}.json")

        if os.path.isfile(base_json):
            filtered_json = base_json
        elif os.path.isfile(alt_json):
            filtered_json = alt_json
        else:
            print(f"[WARN] Rule file not found: {base_json} or {alt_json}, skipping.")
            continue

        valid_indexes = load_rule_indexes(filtered_json)
        if not valid_indexes:
            print(f"[INFO] No valid rule_index in {filtered_json}, skipping.")
            continue

        grounding_subdir = os.path.join(original_grounding_dir, f"{rule_type}_grounding")
        if not os.path.isdir(grounding_subdir):
            print(f"[WARN] Grounding subdirectory not found: {grounding_subdir}, skipping.")
            continue

        out_subdir = os.path.join(output_dir, f"{rule_type}_grounding")
        os.makedirs(out_subdir, exist_ok=True)

        # Check for empty TSV files
        empty_rule_indexes = set()
        for fname in os.listdir(grounding_subdir):
            if fname.endswith(".tsv"):
                parsed_index = parse_rule_index_from_filename(fname)
                if parsed_index and parsed_index in valid_indexes:
                    fpath = os.path.join(grounding_subdir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        if len(lines) <= 1:
                            empty_rule_indexes.add(parsed_index)
                            print(f"[INFO] TSV file {fname} is empty, filtering rule_index: {parsed_index}")
                    except Exception as e:
                        print(f"[WARN] Failed to read file {fname}: {e}")
                        empty_rule_indexes.add(parsed_index)

        # Copy valid files
        for fname in os.listdir(grounding_subdir):
            fpath = os.path.join(grounding_subdir, fname)
            if not os.path.isfile(fpath):
                continue

            parsed_index = parse_rule_index_from_filename(fname)
            if parsed_index and parsed_index in valid_indexes and parsed_index not in empty_rule_indexes:
                out_fpath = os.path.join(out_subdir, fname)
                shutil.copyfile(fpath, out_fpath)

    print(f"[INFO] Groundings filtering completed, results saved in: {output_dir}")

if __name__ == "__main__":
    main()