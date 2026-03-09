import argparse
import json
import os


def assign_global_index_to_single_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        rules = json.load(f)

    for i, rule in enumerate(rules):
        rule["rule_index"] = i

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"Assigned global indices to {len(rules)} rules")
    print(f"Output saved to: {output_file}")

    return len(rules)


def main():
    parser = argparse.ArgumentParser(
        description="Assign global indices to rules in JSON file"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input rules JSON file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path with global indices"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return

    total_rules = assign_global_index_to_single_file(args.input, args.output)
    print(f"Successfully processed {total_rules} rules with global indices!")


if __name__ == "__main__":
    main()