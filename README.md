# BRINK: Benchmark for Reasoning under Incomplete Knowledge

BRINK is the benchmark introduced in our paper **"What Breaks Knowledge Graph based RAG? Empirical Insights into Reasoning under Incomplete Knowledge"**. It is designed to evaluate Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) systems under **knowledge incompleteness**. Unlike standard KGQA benchmarks, BRINK is constructed so that benchmark questions cannot be answered by directly retrieving a single explicit supporting triple. Instead, the answer must be inferred from alternative reasoning paths that remain in the graph after the directly supporting fact is removed.

📄 **Paper:** https://arxiv.org/abs/2508.08344

## Hugging Face Datasets

The released BRINK benchmark datasets are available on Hugging Face:

- **BRINK-family**: https://huggingface.co/datasets/ZDZR/BRINK-family
- **BRINK-FB15k-237**: https://huggingface.co/datasets/ZDZR/BRINK-FB15k-237
- **BRINK-Wikidata5m**: https://huggingface.co/datasets/ZDZR/BRINK-Wikidata5m

Each dataset provides:
- a **complete KG** version
- an **incomplete KG** version, where selected directly supporting triples are removed while preserving alternative reasoning evidence
- train / validation / test question splits
- answer annotations for standardized evaluation

## Dataset Statistics

| Dataset | #Triples | Train | Val | Test | Total Qs |
|---|---:|---:|---:|---:|---:|
| Family-Complete | 17,615 | 1,749 | 218 | 198 | 2,165 |
| Family-Incomplete | 15,785 | 1,749 | 218 | 198 | 2,165 |
| FB15k-237-Complete | 204,087 | 4,374 | 535 | 540 | 5,449 |
| FB15k-237-Incomplete | 198,183 | 4,374 | 535 | 540 | 5,449 |
| Wikidata5m-Complete | 20,510,107 | 27,720 | 3,466 | 3,465 | 34,651 |
| Wikidata5m-Incomplete | 20,478,006 | 27,720 | 3,466 | 3,465 | 34,651 |

## Standardized Evaluation

BRINK includes a standardized evaluation protocol for KG-RAG under incomplete knowledge.

Let the predicted answer set for question `q` be `Pq`, and the gold answer set be `Aq`. We report:

- **Hits@Any**: whether `Pq ∩ Aq ≠ ∅`
- **Precision**: the fraction of predicted answers that are correct
- **Recall**: the fraction of gold answers that are predicted
- **F1**: the per-question harmonic mean based on the predicted and gold answer sets
- **Hits@Hard**: whether the prediction contains the **hard answer**, i.e. the specific answer whose directly supporting triple was intentionally removed during benchmark construction
- **HHR (Hard Hits Rate)**: `Hits@Hard / Hits@Any`

## Prediction Input Requirement

The official evaluation script expects the **model's raw output text**, without any user-side manual or model-specific postprocessing.

This means:
- pass the model's **original generated output** directly to the evaluation script
- do **not** manually rewrite, filter, rerank, or normalize predictions before evaluation
- output normalization and answer extraction are handled inside the evaluator for fair comparison across systems

## Evaluation Script

We recommend evaluating model predictions using the official script:

```bash
python evaluation/evaluate_brink.py \
  --gold path/to/gold.json \
  --pred path/to/predictions.json
```

## Prediction Input Requirement

The official evaluation script expects the **model's raw output string** as input, without any user-side manual or model-specific postprocessing.

This means:
- pass the model's **original generated output** directly to the evaluation script
- do **not** manually rewrite, filter, rerank, or normalize predictions before evaluation

Inside the evaluator, the raw output string is converted into a prediction set using the official BRINK postprocessing procedure. Following the paper, the evaluator first applies a splitting function to convert the raw output string into candidate answers, and then applies normalization before matching predictions against gold answers.

### Prediction File Format

The prediction file should contain one **raw model output string** per question.

Example:

```json
[
  {
    "id": "q1",
    "raw_output": "Paris, London"
  },
  {
    "id": "q2",
    "raw_output": "Marriage"
  }
]
```

### Gold File Format

The gold file should contain:
- question id
- question text
- gold answer set
- hard answer

Example:

```json
[
  {
    "id": "q1",
    "question": "Which cities are associated with Country X in the benchmark?",
    "answers": ["Paris", "London"],
    "hard_answer": "Paris"
  },
  {
    "id": "q2",
    "question": "What is the relationship between Person A and Person B?",
    "answers": ["Marriage"],
    "hard_answer": "Marriage"
  }
]
```
### Example Files

Example gold and prediction files are provided in:

- `evaluation/examples/example_gold.json`
- `evaluation/examples/example_pred.json`

You can test the evaluator with:

```bash
python evaluation/evaluate_brink.py \
  --gold evaluation/examples/example_gold.json \
  --pred evaluation/examples/example_pred.json
```

## Installation

```bash
pip install -r requirements.txt
pip install openai --upgrade
```

### Java Requirements

- Java 9.0 or higher is required for AMIE3
- Download AMIE3 from: https://github.com/dig-team/amie/releases/tag/v3.5.1

## Pipeline Overview

The benchmark construction pipeline follows these main steps:

1. **Rule Mining** - extract logical rules using AMIE3
2. **Grounding Generation** - generate rule groundings with index tracking
3. **Triple Removal** - remove direct evidence while preserving reasoning paths
4. **Question Generation** - create natural language questions from groundings
5. **Answer Completion** - compute complete answer sets for evaluation

## Dataset Pipeline Details

The following commands describe the benchmark construction pipeline for Family, FB15k-237, and Wikidata5m.

### 1. Rule Mining

Extract rules with different maximum atom depths:

```bash
java -jar amie3.5.1.jar -maxad 3 -minhc 0.1 -minc 0.3 -minpca 0.4 - data/family/facts.tsv > amie_output_family.txt
java -jar amie3.5.1.jar -maxad 4 -minhc 0.1 -minc 0.3 -minpca 0.4 - data/FB15K-237_Private_ID/facts.tsv > amie_output_fb15k237.txt
java -jar amie3.5.1.jar -maxad 4 -minhc 0.1 -minc 0.3 -minpca 0.4 - data/Wikidata5m_Private_ID/facts.tsv > amie_output_wikidata5m.txt
```

### Rule Processing

Convert AMIE output to structured format and assign global unique indices to rules:

```bash
sh ./scripts/convert_amie_txt_to_json.sh
```

### 2. Grounding Generation

Generate all rule groundings with automatic empty filtering:

```bash
sh ./scripts/grounding_generation.sh
```

### Grounding Sampling

Sample up to 30 groundings per rule and filter duplicates:

```bash
sh ./scripts/grounding_sample.sh
```

### Data Splitting

Split groundings before question generation for efficiency:

```bash
sh ./scripts/data_splitting.sh
```

### 3. Triple Removal

Create incomplete knowledge graphs by removing direct evidence:

```bash
sh ./scripts/triple_removal.sh
```

### 4. Question Generation

Generate questions with resume capability:

```bash
sh ./scripts/question_generation.sh
```

### Dataset Balancing

Balance answer distributions:

```bash
sh ./scripts/auto_downsample_answers.sh
```

### Answer Set Completion

Generate complete answer sets for each question using the original KG:

```bash
sh ./scripts/complete_answer_search.sh
```

## Recommended Evaluation Workflow for New Models

To evaluate any new KG-RAG system on BRINK:

1. Load the corresponding BRINK dataset split.
2. Run your model on each question using the provided KG setting.
3. Save the **raw model output text** for each question in the prediction JSON format described above.
4. Run the official evaluation script to obtain standardized metrics.

This design ensures that all systems are compared under the same output-processing and metric definitions.


## Citation

If you use BRINK, please cite:

```bibtex
@inproceedings{zhou2026breaks,
  title={What Breaks Knowledge Graph based RAG? Benchmarking and Empirical Insights into Reasoning under Incomplete Knowledge},
  author={Zhou, Dongzhuoran and Zhu, Yuqicheng and Wang, Xiaxia and Zhou, Hongkuan and He, Yuan and Chen, Jiaoyan and Staab, Steffen and Kharlamov, Evgeny},
  booktitle={The 19th Conference of the European Chapter of the Association for Computational Linguistics: EACL},
  year={2026}
}
```

## License

Please refer to the repository license for usage terms.
