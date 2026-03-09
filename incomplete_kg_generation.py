import os
import argparse
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from openai import AzureOpenAI
from tqdm.auto import tqdm
import csv
import ast

_ = load_dotenv(find_dotenv())

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(exclude_managed_identity_credential=True),
    "https://cognitiveservices.azure.com/.default"
)
client = AzureOpenAI(azure_ad_token_provider=token_provider)

def ensure_list_string(value):
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return value
    except Exception:
        pass
    return str([value]) if value else "[]"

def postprocess_entity_fields(input_path, output_path):
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        for key in ["q_entity", "answer", "a_entity"]:
            row[key] = ensure_list_string(row.get(key, ""))

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Post-processed file saved to: {output_path}")

def safe_extract_first(item):
    try:
        parsed = ast.literal_eval(item)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
    except:
        pass
    return item.strip()

def clean_output_entity_check(input_path, output_path):
    cleaned_rows = []
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        headers = reader.fieldnames
        for row in reader:
            question = row.get("question", "").strip()

            raw_q = row.get("q_entity", "").strip()
            try:
                q_list = ast.literal_eval(raw_q) if raw_q else []
            except Exception:
                q_list = []
            q_ent = q_list[0] if q_list else ""

            raw_a = row.get("a_entity", "").strip()
            try:
                a_list = ast.literal_eval(raw_a) if raw_a else []
            except Exception:
                a_list = []
            a_ent = a_list[0] if a_list else ""

            if question and q_ent and q_ent in question and (not a_ent or a_ent not in question):
                cleaned_rows.append(row)

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"Cleaned valid questions saved to: {output_path}")
    return len(cleaned_rows)

def clean_output(input_path, output_path):
    seen_ids = set()
    cleaned_rows = []
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        headers = reader.fieldnames
        for row in reader:
            rec_id = row["id"].strip()
            if rec_id not in seen_ids:
                seen_ids.add(rec_id)
                cleaned_rows.append(row)
    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"Deduplicated by ID. {len(cleaned_rows)} unique rows saved to: {output_path}")
    return len(cleaned_rows)

def sort_by_id(input_path, output_path):
    """Sort file by ID from small to large."""
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames

    try:
        sorted_rows = sorted(rows, key=lambda x: int(x["id"]))
        print(f"Sorted {len(sorted_rows)} rows by ID (integer)")
    except ValueError:
        sorted_rows = sorted(rows, key=lambda x: x["id"])
        print(f"Sorted {len(sorted_rows)} rows by ID (string)")

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Sorted file saved to: {output_path}")
    return len(sorted_rows)

def select_instances(df, n, method="top"):
    if method == "random":
        return df.sample(n, random_state=42)
    elif method == "top":
        if "support" in df.columns:
            return df.sort_values(by="support", ascending=False).head(n)
        else:
            return df.head(n)
    else:
        raise ValueError("Selection method must be 'random' or 'top'.")

def parse_grounding(grounding_str):
    if "=>" not in grounding_str:
        raise ValueError("Grounding string missing '=>' separator.")
    _, head_str = grounding_str.split("=>", 1)
    parts = head_str.strip().split()
    if len(parts) < 3:
        raise ValueError("Head grounding must have at least 3 tokens.")
    return {
        "entity_x": parts[0],
        "relation_T": parts[1],
        "entity_z": parts[2],
    }

def generate_qa_openai(instance, prompt_template):
    grounding_info = parse_grounding(instance["grounding"])
    prompt = prompt_template.format(
        entity_x=grounding_info["entity_x"],
        relation_T=grounding_info["relation_T"],
        entity_z=grounding_info["entity_z"],
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert KGQA question generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    answer_text = response.choices[0].message.content.strip()

    qa = {"question": "", "q_entity": "", "a_entity": ""}
    for line in answer_text.splitlines():
        line = line.strip()
        if line.lower().startswith("question:"):
            qa["question"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("question entity:"):
            qa["q_entity"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer entity:"):
            qa["a_entity"] = line.split(":", 1)[1].strip()
    return qa

def get_prompt_template(prompt_type):
    if prompt_type == "general":
        return GENERAL_PROMPT_TEMPLATE
    else:
        raise ValueError("Unknown prompt type")

GENERAL_PROMPT_TEMPLATE = """
You are an expert in knowledge graph question generation.
A triple has been removed from a knowledge graph:

Removed Triple: ({entity_x}, {relation_T}, {entity_z})

Note: The numbers in the entities (e.g., {entity_x}, {entity_z}) represent specific individual identifiers.

Your task is to generate the following outputs:
1. Question Entity: Exactly one of the entities from the removed triple (either {entity_x} or {entity_z} exactly, with no additions).
2. Answer Entity: The remaining entity from the removed triple. That is, if you choose {entity_x} as the Question Entity, then the Answer Entity must be {entity_z}, and vice versa.
3. Question: A clear, natural-language question asking for the Answer Entity from the removed triple. The question must include the given relation {relation_T} in a way that adheres to the following constraint:
   - While you may omit or adjust auxiliary parts of {relation_T} (for example, you may remove the suffix "_of"), the core word (the "xxx" in the "xxx_of" structure) must remain unchanged.
   - The relation {relation_T} should be expressed naturally in your question; you may paraphrase it instead of using the raw {relation_T}, as long as the core meaning (the "xxx" in "xxx_of") is preserved (for example, you can convert "born_in" to "was born in").

The question must include the Question Entity. The question must not include the Answer Entity.
The question must not be a simple yes/no question; it should require an explicit answer.

Please output exactly three lines, in plain text, without any Markdown formatting.
Do NOT use asterisks (*), bullets, numbering, or any other markup.

Example:
Removed Triple: ("Alice", "wife_of", "Carol").
Output:
Question: Who is Carol's wife?
Question Entity: Carol
Answer Entity: Alice

Now, generate the outputs for:
Removed Triple: ({entity_x}, {relation_T}, {entity_z})
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--model_choice", choices=["openai", "llama"], default="openai")
    parser.add_argument("--prompt_type", choices=["general", "family"], default="general")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep="\t")
    if "id" not in df.columns:
        raise ValueError("Input file must contain an 'id' column.")
    df["id"] = df["id"].astype(str)

    # Clean duplicate IDs from input and save
    if df["id"].duplicated().any():
        num_dups = df["id"].duplicated().sum()
        print(f"[CLEAN] Found {num_dups} duplicate IDs. Keeping first occurrences.")
        df = df[~df["id"].duplicated(keep="first")]
        df.to_csv(args.input_file, sep="\t", index=False)

    # Resume logic - load processed IDs
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                processed_ids.add(row["id"])

    df_unprocessed = df[~df["id"].isin(processed_ids)]
    if df_unprocessed.empty:
        print("All instances already processed.")
        return

    df_selected = select_instances(df_unprocessed, min(args.n_instances, len(df_unprocessed)))
    prompt_template = get_prompt_template(args.prompt_type)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_header = not os.path.exists(args.output_file)

    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} exists. Cleaning it before appending new data.")
        clean_output(args.output_file, args.output_file)

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                processed_ids.add(row["id"])

    with open(args.output_file, "a", newline='', encoding="utf-8") as fout:
        fieldnames = ["id", "rule_index", "question", "q_entity", "answer", "a_entity", "rule", "grounding"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        flush_every = 20
        for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Generating QA"):
            inst = row.to_dict()

            for field in ["id", "rule_index", "rule", "grounding"]:
                if field not in inst:
                    raise KeyError(f"Missing required field: {field}")

            rec_id = inst["id"]
            if rec_id in processed_ids:
                continue

            if args.model_choice == "openai":
                qa = generate_qa_openai(inst, prompt_template)
            else:
                raise NotImplementedError("LLaMA not implemented.")

            question = qa["question"]
            q_ent = qa["q_entity"]
            a_ent = qa["a_entity"]

            if q_ent and q_ent not in question:
                print(f"[WARN] q_entity `{q_ent}` not in question: {question}")
                question = ""
            if a_ent and a_ent in question:
                print(f"[WARN] a_entity `{a_ent}` appears in question: {question}")
                question = ""

            writer.writerow({
                "id": rec_id,
                "rule_index": inst["rule_index"],
                "question": question,
                "q_entity": [q_ent] if q_ent else [],
                "answer": [a_ent] if a_ent else [],
                "a_entity": [a_ent] if a_ent else [],
                "rule": [inst["rule"]] if inst["rule"] else [],
                "grounding": [inst["grounding"]] if inst["grounding"] else []
            })

            if idx % flush_every == 0:
                fout.flush()
            processed_ids.add(rec_id)

    print(f"QA dataset saved to {args.output_file}")

    postprocess_entity_fields(args.output_file, args.output_file)
    print(f"Postprocessed output saved to: {args.output_file}")

    cleaned_output_path = (
        args.output_file if args.output_file.endswith("_cleaned.tsv")
        else args.output_file.replace(".tsv", "_cleaned.tsv")
    )

    clean_output(args.output_file, cleaned_output_path)
    print(f"Cleaned output saved to {cleaned_output_path}")

    args.output_file = cleaned_output_path

    final_cleaned_path = cleaned_output_path.replace(".tsv", "_final.tsv")
    clean_output_entity_check(cleaned_output_path, final_cleaned_path)
    print(f"Final cleaned output saved to: {final_cleaned_path}")

    sort_by_id(final_cleaned_path, final_cleaned_path)
    print(f"Final output sorted by ID: {final_cleaned_path}")

if __name__ == "__main__":
    main()