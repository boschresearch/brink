import pandas as pd
import ast
import argparse

def extract_head_from_grounding(grounding_str):
    try:
        groundings = ast.literal_eval(grounding_str)
        if not groundings:
            return None
        last = groundings[0].split("=>")[-1].strip()
        tokens = last.split()
        if len(tokens) != 3:
            return None
        return tokens[0], tokens[1], tokens[2]
    except:
        return None

def find_matching_answers(row, facts_df):
    try:
        head_triple = extract_head_from_grounding(row["grounding"])
        if not head_triple:
            return []
        head_entity = str(head_triple[0])
        relation = head_triple[1]

        matches = facts_df[
            (facts_df["head"].astype(str) == head_entity) &
            (facts_df["relation"] == relation)
        ]
        return matches["tail"].astype(str).unique().tolist()
    except Exception as e:
        print(f"Error in row {row.get('id', '')}: {e}")
        return []

def main(args):
    facts_df = pd.read_csv(args.facts_file, sep="\t", header=None, names=["head", "relation", "tail"])
    test_df = pd.read_csv(args.test_file, sep="\t")

    test_df["completed_a_entity"] = test_df.apply(lambda row: find_matching_answers(row, facts_df), axis=1)

    # Replace a_entity and answer
    test_df["a_entity"] = test_df["completed_a_entity"]
    test_df["answer"] = test_df["completed_a_entity"]

    # Save output
    output_df = test_df.drop(columns=["completed_a_entity"])
    output_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Output saved with replaced a_entity and answer: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="complete_answer.tsv")
    args = parser.parse_args()
    main(args)