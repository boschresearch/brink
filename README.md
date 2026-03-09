# What Breaks Knowledge Graph based RAG? Empirical Insights into Reasoning under Incomplete Knowledge
A framework for constructing benchmarks to evaluate Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) systems under knowledge incompleteness conditions.

# Installation
```bash
pip install -r requirements.txt
pip install openai --upgrade
```

**Java Requirements:**
- Java 9.0 or higher is required for AMIE3
- Download AMIE3 from: https://github.com/dig-team/amie/releases/tag/v3.5.1

## Pipeline Overview

The benchmark construction follows these main steps:
1. **Rule Mining** - Extract logical rules using AMIE3
2. **Grounding Generation** - Generate rule groundings with index tracking
3. **Triple Removal** - Remove direct evidence while preserving reasoning paths
4. **Question Generation** - Create natural language questions from groundings

## Dataset Pipeline Details (Family, FB15k-237)

### 1. Rule Mining 
Extract rules with different maximum atom depths:
```bash
java -jar amie3.5.1.jar -maxad 3 -minhc 0.1 -minc 0.3  -minpca 0.4  - data/family/facts.tsv > amie_output_family.txt
java -jar amie3.5.1.jar -maxad 4 -minhc 0.1 -minc 0.3 -minpca 0.4  - data/FB15K-237_Private_ID/facts.tsv > amie_output_fb15k237.txt
java -jar amie3.5.1.jar -maxad 4 -minhc 0.1 -minc 0.3 -minpca 0.4  - data/Wikidata5m_Private_ID/facts.tsv > amie_output_wikidata5m.txt
```


#### Rule Processing
Convert AMIE output to structured format and assign global unique indices to rules:
```bash
sh ./scripts/convert_amie_txt_to_json.sh
```
### 2. Grounding Generation
Generate all groundings from rules with automatic empty filtering:
```bash
sh ./scripts/grounding_generation.sh
```

#### Grounding Sampling
Sample up to 30 groundings per rule and filter duplicates:
```bash
sh ./scripts/grounding_sample.sh
```

#### Data Splitting
Split groundings before question generation for efficiency:
```bash
sh ./scripts/data_splitting.sh
```

### 3. Triple Removal
Create incomplete knowledge graph by removing direct evidence:
```bash
sh ./scripts/triple_removal.sh
```

### 4. Question Generation

Generate questions with resume capability:
```bash
sh ./scripts/question_generation.sh
```

#### Dataset Balancing
Balance answer distribution
```bash
sh ./scripts/auto_downsample_answers.sh
```
#### Answer Set Completion
Generate complete answer sets for each question using the original KG:
```bash
sh ./scripts/complete_answer_search.sh
```
