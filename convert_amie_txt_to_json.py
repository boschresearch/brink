import sys
import json


def is_rule_line(line):
    return line.startswith('?') and '=>' in line


def convert_amie_txt_to_json(input_file, output_file):
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = [line.strip() for line in fin if line.strip()]
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='utf-16') as fin:
            lines = [line.strip() for line in fin if line.strip()]

    # Find header line index
    header_idx = 0
    for i, line in enumerate(lines):
        if "Head Coverage" in line:
            header_idx = i
            break

    # Filter rule lines after header
    rule_lines = []
    for line in lines[header_idx + 1:]:
        if is_rule_line(line):
            rule_lines.append(line)

    # Parse each rule line (last 7 fields are numeric, rest is rule)
    for line in rule_lines:
        tokens = line.split()
        if len(tokens) < 8:
            continue
        rule_tokens = tokens[:-7]
        rule_str = " ".join(rule_tokens)
        try:
            head_coverage = float(tokens[-7])
        except:
            head_coverage = tokens[-7]
        try:
            std_confidence = float(tokens[-6])
        except:
            std_confidence = tokens[-6]
        try:
            pca_confidence = float(tokens[-5])
        except:
            pca_confidence = tokens[-5]
        positive_examples = tokens[-4]
        body_size = tokens[-3]
        pca_body_size = tokens[-2]
        functional_variable = tokens[-1]

        rule_dict = {
            "rule": rule_str,
            "head_coverage": head_coverage,
            "std_confidence": std_confidence,
            "pca_confidence": pca_confidence,
            "positive_examples": positive_examples,
            "body_size": body_size,
            "pca_body_size": pca_body_size,
            "functional_variable": functional_variable
        }
        data.append(rule_dict)

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print("Converted {} rules to JSON format.".format(len(data)))
    print("saved to", output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_amie_to_json.py input.txt output.json")
    else:
        convert_amie_txt_to_json(sys.argv[1], sys.argv[2])