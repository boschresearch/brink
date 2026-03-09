#!/usr/bin/env python3
import argparse
import csv
import ast
import os
import glob
import random

def sample_groundings_from_file(file_path, n_instances):
    """Sample first n_instances from a rule file (skip header)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []
    return lines[1:1 + n_instances]

def combine_sampled_groundings_single_dir(input_dir, max_rules, n_instances, random_seed=42):
    """
    From all rule_*.tsv files in directory:
      - Randomly select max_rules files (if specified)
      - Sample n_instances records from each file
      - Combine all sampled records
    """
    combined_rows = []
    header = None

    rule_files = sorted(glob.glob(os.path.join(input_dir, 'rule_*.tsv')))

    if not rule_files:
        print(f"Warning: No rule_*.tsv files found in {input_dir}")
        return combined_rows

    if max_rules and max_rules < len(rule_files):
        random.seed(random_seed)
        rule_files = random.sample(rule_files, max_rules)
        print(f"Randomly selected {len(rule_files)} from {len(glob.glob(os.path.join(input_dir, 'rule_*.tsv')))} files")

    for file in rule_files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            file_rows = list(reader)
            if not file_rows:
                continue

            if header is None:
                header = file_rows[0]
                combined_rows.append(header)

            sample = file_rows[1:1 + n_instances]
            combined_rows.extend(sample)

        print(f"Sampled {min(n_instances, len(file_rows) - 1)} records from {os.path.basename(file)}")

    return combined_rows

def parse_single_triple(triple_str):
    parts = triple_str.split()
    if len(parts) == 3:
        return tuple(parts)
    return None

def parse_grounding(grounding_str):
    """Parse grounding string, return (head_triples, body_triples)."""
    head_triples = []
    body_triples = []
    if not grounding_str.strip():
        return head_triples, body_triples

    segments = grounding_str.split(',')
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if '=>' in seg:
            body_part, head_part = seg.split('=>', 1)
            body_part = body_part.strip()
            head_part = head_part.strip()
            body_items = body_part.split('|')
            for item in body_items:
                t = parse_single_triple(item.strip())
                if t:
                    body_triples.append(t)
            head_items = head_part.split('|')
            for item in head_items:
                t = parse_single_triple(item.strip())
                if t:
                    head_triples.append(t)
        else:
            body_items = seg.split('|')
            for item in body_items:
                t = parse_single_triple(item.strip())
                if t:
                    body_triples.append(t)
    return head_triples, body_triples

def filter_groundings(rows):
    """Filter out rows where head triples appear in other rows' body triples."""
    header = rows[0]
    try:
        grounding_idx = header.index("grounding")
    except ValueError:
        print("Header does not contain 'grounding' column.")
        return rows

    n = len(rows) - 1
    body_triples_list = []
    head_triples_list = []
    for row in rows[1:]:
        grounding_str = row[grounding_idx]
        # Try to parse if grounding is a list string
        try:
            grounding_eval = ast.literal_eval(grounding_str)
            if isinstance(grounding_eval, list) and grounding_eval:
                grounding_str = grounding_eval[0]
        except Exception:
            pass
        head_triples, body_triples = parse_grounding(grounding_str)
        body_set = set(" ".join(t) for t in body_triples)
        head_set = set(" ".join(t) for t in head_triples)
        body_triples_list.append(body_set)
        head_triples_list.append(head_set)

    filtered_rows = [rows[0]]
    for i in range(n):
        current_head = head_triples_list[i]
        other_bodies = set()
        for j in range(n):
            if i == j:
                continue
            other_bodies.update(body_triples_list[j])
        if current_head.intersection(other_bodies):
            continue
        else:
            filtered_rows.append(rows[i + 1])
    return filtered_rows

def main():
    parser = argparse.ArgumentParser(
        description="Sample and filter grounding data from single grounding directory"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing all rule_*.tsv files")
    parser.add_argument("--max_rules", type=int, default=None,
                        help="Maximum number of rule files to select (random)")
    parser.add_argument("--n_instances", type=int, default=30,
                        help="Number of groundings to sample from each rule file")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_file", required=True,
                        help="Output path for combined and filtered groundings (TSV)")
    args = parser.parse_args()

    combined_rows = combine_sampled_groundings_single_dir(
        args.input_dir, args.max_rules, args.n_instances, args.random_seed
    )

    if not combined_rows:
        print("No grounding files found.")
        return

    print("Total rows after sampling (including header):", len(combined_rows))
    filtered_rows = filter_groundings(combined_rows)
    print("Total rows after filtering (including header):", len(filtered_rows))

    # Add example_id to filtered examples
    header = ["id", "rule_index", "rule", "grounding"]
    indexed_rows = [header]
    for idx, row in enumerate(filtered_rows[1:]):
        new_row = [str(idx)] + row
        indexed_rows.append(new_row)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerows(indexed_rows)

    print("Filtered groundings saved to", args.output_file)
    print(f"Final statistics: {len(indexed_rows) - 1} valid grounding records")

if __name__ == "__main__":
    main()