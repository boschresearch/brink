import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

def normalize_token(token):
    return token.strip().strip("\"'")

def load_facts(fact_file):
    fact_list = []
    pred_to_facts = {}
    facts_set = set()
    with open(fact_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triple = tuple(map(normalize_token, parts))
            fact_list.append(triple)
            facts_set.add(triple)
            pred_to_facts.setdefault(triple[1], []).append(triple)
    return fact_list, pred_to_facts, facts_set

def parse_rule_tokens(rule_str):
    if "=>" not in rule_str:
        return [], []
    left, right = rule_str.split("=>", 1)
    left_tokens = left.strip().split()
    right_tokens = right.strip().split()
    body = [tuple(left_tokens[i:i + 3]) for i in range(0, len(left_tokens), 3)]
    head = [tuple(right_tokens[i:i + 3]) for i in range(0, len(right_tokens), 3)]
    return body, head

def load_rules(rules_file):
    rules = []
    with open(rules_file, "r") as f:
        raw = json.load(f)
    for entry in raw:
        if not isinstance(entry, dict) or "rule" not in entry:
            continue
        rule_str = entry["rule"]
        body, head = parse_rule_tokens(rule_str)
        if not body or not head:
            continue
        rules.append({
            "rule": rule_str,
            "body": body,
            "head": head,
            "meta": entry
        })
    return rules

def match_atom(atom, fact, binding):
    new_binding = dict(binding)
    for token, value in zip(atom, fact):
        if token.startswith("?"):
            if token in new_binding and new_binding[token] != value:
                return None
            new_binding[token] = value
        elif token != value:
            return None
    return new_binding

def match_atom_with_facts(atom, pred_to_facts, binding):
    subj, pred, obj = atom
    candidates = pred_to_facts.get(pred, [])
    if subj.startswith("?") and subj in binding:
        candidates = [f for f in candidates if f[0] == binding[subj]]
    elif obj.startswith("?") and obj in binding:
        candidates = [f for f in candidates if f[2] == binding[obj]]
    for fact in candidates:
        new_bind = match_atom(atom, fact, binding)
        if new_bind is not None:
            yield new_bind

def match_body_atoms(body_atoms, pred_to_facts, binding):
    if not body_atoms:
        yield binding
        return
    first, rest = body_atoms[0], body_atoms[1:]
    for new_binding in match_atom_with_facts(first, pred_to_facts, binding):
        yield from match_body_atoms(rest, pred_to_facts, new_binding)

def apply_binding(atoms, binding):
    return [tuple(binding.get(x, x) if x.startswith("?") else x for x in atom) for atom in atoms]

def atoms_to_str(atoms):
    return " | ".join(" ".join(atom) for atom in atoms)

def process_single_rule(rule_obj, pred_to_facts, facts_set, output_dir):
    rule_str = rule_obj["rule"]
    body, head = rule_obj["body"], rule_obj["head"]
    rule_index = rule_obj["meta"].get("rule_index", "unknown")
    bindings = list(match_body_atoms(body, pred_to_facts, {}))

    valid_groundings = []
    for b in bindings:
        grounded_body = apply_binding(body, b)
        grounded_head = apply_binding(head, b)
        if any(h in grounded_body for h in grounded_head):
            continue
        if not all(triple in facts_set for triple in grounded_body + grounded_head):
            continue
        valid_groundings.append((grounded_body, grounded_head))

    if not valid_groundings:
        print(f"Rule {rule_index} has no valid groundings, skipping file creation")
        return {
            "rule_index": rule_index,
            "rule": rule_str,
            "num_groundings": 0,
            "files_created": False
        }

    # JSON output
    rule_output = {
        "rule_index": rule_index,
        "rule": rule_str,
        "groundings": [{"body": b, "head": h} for b, h in valid_groundings]
    }
    json_path = os.path.join(output_dir, f"rule_{rule_index}_groundings.json")
    with open(json_path, "w") as f:
        json.dump(rule_output, f, indent=2)

    # TSV output
    tsv_path = os.path.join(output_dir, f"rule_{rule_index}_groundings.tsv")
    with open(tsv_path, "w") as f:
        f.write("rule_index\trule\tgrounding\n")
        for b, h in valid_groundings:
            f.write(f"Rule_{rule_index}\t{rule_str}\t{atoms_to_str(b)} => {atoms_to_str(h)}\n")

    return {
        "rule_index": rule_index,
        "rule": rule_str,
        "num_groundings": len(valid_groundings),
        "files_created": True
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_rules", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    facts, pred_to_facts, facts_set = load_facts(args.facts)
    rules = load_rules(args.rules)

    if args.random_seed is not None:
        random.seed(args.random_seed)
    if args.max_rules:
        max_n = min(args.max_rules, len(rules))
        rules = random.sample(rules, max_n)

    tasks = [(r, pred_to_facts, facts_set, args.output) for r in rules]
    results = []
    files_created = 0
    rules_with_groundings = 0

    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.starmap(process_single_rule, tasks), total=len(tasks), desc="Processing Rules"):
            if result["files_created"]:
                files_created += 1
                print(f"Rule {result['rule_index']} matched {result['num_groundings']} groundings")
            else:
                print(f"Rule {result['rule_index']} skipped (no groundings)")

            if result["num_groundings"] > 0:
                rules_with_groundings += 1

            results.append(result)

    # Save summary
    summary = {
        "total_rules_processed": len(results),
        "rules_with_groundings": rules_with_groundings,
        "files_created": files_created,
        "rules_skipped": len(results) - files_created,
        "results": results
    }

    with open(os.path.join(args.output, "grounding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"   Total rules processed: {len(results)}")
    print(f"   Rules with groundings: {rules_with_groundings}")
    print(f"   Files created: {files_created}")
    print(f"   Rules skipped (empty): {len(results) - files_created}")

if __name__ == "__main__":
    main()