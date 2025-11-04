# On Bidirectional Reasoning of LLMs for Software Engineering

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code, data, and experimental results for the paper **"On Bidirectional Reasoning of LLMs for Software Engineering"**. This research investigates whether fine-tuned Large Language Models (LLMs) genuinely understand their specialized tasks or merely replicate statistical patterns.

## 🎯 Research Question

**When fine-tuned on code transformation tasks, do LLMs genuinely understand what they are specialized to perform, or do they merely replicate statistical patterns?**

We propose **bidirectional reasoning** as a test: if a model truly understands a task it was fine-tuned on, it should be able to perform its inverse without explicit reverse fine-tuning.

![Workflow Diagram](workflow.png)

## 📦 Reproducibility

**All code for complete reproduction of experiments is provided in this repository.**

**Note on Data Availability:**
- Due to GitHub repository size limitations, we provide **sample training datasets** (1K examples each) rather than the full 10,000-example datasets used in the paper.
- **Test case execution reports** (hundreds of individual test reports) are not included due to their large size.
- All evaluation scripts and code for dataset generation are included, enabling full reproduction of all experiments and results.


## 📁 Repository Structure

```
Bidirectional_Study/
│
├── Experiment1/                     # Standard Fine-Tuning (SFT & BFT)
│   ├── API_Model_fine_tuning.py     # OpenAI API fine-tuning
│   ├── open-source-model-fine-tuning.py  # LoRA fine-tuning 
│   ├── Auto-Correction_Loop.py      # Iterative correction
│   ├── compare_codebleu.py          # CodeBLEU evaluation
│   ├── Test_Cases_Evaluation.py     # Functional correctness testing
│   └── Fine-Tuning_Datasets/        # Training datasets
│       ├── VariableRenaming_Dataset_1K.jsonl
│       ├── Deadcode_Dataset_1K.jsonl
│       └── IntStringEncryption_Dataset_1K.jsonl
│
├── Experiment2/                     # Prompting Strategies & Readability
│   ├── generate_deobfuscated_Code_Prompting_Strategy.py
│   ├── Readability_Computation.py
│   ├── readability.classifier      # Readability classifier model
│   └── README.md                    # Experiment-specific documentation
│
├── Experiment3-Contrastive-Fine-tuning/  # Contrastive Fine-Tuning (CFT)
│   ├── Contrastive_Fine-tuning.py   # API-based CFT
│   ├── Open-source-Model_Contrastive_FT.py
│   └── Contrastive_Traing_Datasets/ # CFT training datasets
│       ├── csv_variable_renaming_fine_tuning.jsonl
│       ├── deadcode_fine_tuning.jsonl
│       └── int&string_encryption.jsonl
│
├── Evaluation_Datasets/             # Evaluation datasets
│   ├── Original_Evaluation_code_bytool.json
│   ├── Variable-renaming_Obfuscator_Tool.json
│   ├── Deadcode_Obfuscator_Tool.json
│   └── Literals-Encrypt_Obfuscator_Tool.json
│
└── README.md                         # This file
```

**Note**: 
- Experiment 2 (Multi-variable Training) has been removed from the latest version of the paper and is deprecated.
- The repository structure matches the experimental setup described in the paper.

## 🔬 Experiments

### Experiment 1: Standard Fine-Tuning (SFT)
**Location**: `Experiment1/`

Evaluates baseline fine-tuning performance on forward obfuscation tasks:
- **Models**: 6 LLMs (4 open-source: Qwen2.5-7B, Qwen2.5-Coder-7B, DeepSeek-R1.5-7B, Mistral-7B; 2 API: GPT-3.5-Turbo, GPT-4.1-mini)
- **Training**: 10,000 pairs (open-source) / 1,000 pairs (API) per transformation
- **Transformations**: Variable Renaming, Dead Code Insertion, Literal Encryption
- **Evaluation**: Semantic correctness, CodeBLEU, AST similarity, readability

**Key Scripts**:
- `API_Model_fine_tuning.py`: OpenAI API-based fine-tuning
- `open-source-model-fine-tuning.py`: LoRA fine-tuning for open-source models
- `Auto-Correction_Loop.py`: Iterative code correction for failed generations
- `compare_codebleu.py`: CodeBLEU evaluation
- `Test_Cases_Evaluation.py`: Functional correctness testing

### Experiment 3: Prompting Strategies and Readability
**Location**: `Experiment2/`

Investigates whether advanced prompting strategies can enable reverse reasoning:
- **Prompting Strategies**: Simple prompting, Few-shot learning, Chain-of-Thought (CoT), Augmented reasoning
- **Readability Analysis**: JSEP 2018 Comprehensive Model for Code Readability
- **Finding**: All prompting strategies fail to enable reverse reasoning (0% improvement)

**Key Scripts**:
- `generate_deobfuscated_Code_Prompting_Strategy.py`: Multiple prompting strategies
- `Readability_Computation.py`: Code readability analysis

### Experiment 3: Contrastive Fine-Tuning (CFT)
**Location**: `Experiment4-Contrastive-Fine-tuning/`

Investigates whether contrastive learning can enable emergent bidirectional reasoning:
- **Methodology**: Semantic triplets (anchor, positive, negative) with multi-objective loss
- **Result**: Emergent bidirectionality on lexical transformations (39-52% reverse) but failure on structural/semantic (0%)

**Key Scripts**:
- `Contrastive_Fine-tuning.py`: API-based contrastive fine-tuning
- `Open-source-Model_Contrastive_FT.py`: Open-source model implementation

## 🛠️ Setup and Installation

### Prerequisites

```bash
# Python 3.8+
pip install openai transformers torch datasets peft tqdm nltk lizard pandas scikit-learn
```

### Environment Setup

```bash
# Set OpenAI API key (required for API-based experiments)
export OPENAI_API_KEY="your-api-key-here"
```

### Dataset Preparation

**Important Note on Data Availability:**

Due to GitHub repository size limitations, we provide:
- **Training Datasets**: Sample datasets (1K examples each) in `Experiment1/Fine-Tuning_Datasets/`
  - `VariableRenaming_Dataset_1K.jsonl`: 1,000 examples for variable renaming
  - `Deadcode_Dataset_1K.jsonl`: 1,000 examples for dead code insertion
  - `IntStringEncryption_Dataset_1K.jsonl`: 1,000 examples for literal encryption
  - **Note**: The full 10,000-example datasets used in the paper are not included due to size constraints, but all code for dataset generation and reproduction is provided.

- **Evaluation Datasets**: Complete evaluation datasets in `Evaluation_Datasets/`
  - `Original_Evaluation_code_bytool.json`: Original Java code samples
  - `Variable-renaming_Obfuscator_Tool.json`: Variable renaming obfuscated code
  - `Deadcode_Obfuscator_Tool.json`: Dead code obfuscated code
  - `Literals-Encrypt_Obfuscator_Tool.json`: Literal encryption obfuscated code

- **Test Case Results**: Individual test case execution reports are not included due to their large size (hundreds of test reports per experiment). However, all evaluation scripts (`Test_Cases_Evaluation.py`, `compare_codebleu.py`, etc.) are provided for full reproducibility.

**All code for complete reproduction of experiments is provided in this repository.**

## 🚀 Usage

### Running Standard Fine-Tuning (Experiment 1)

```bash
cd Experiment1

# API-based fine-tuning
python API_Model_fine_tuning.py

# Open-source model fine-tuning (LoRA)
python open-source-model-fine-tuning.py

# Evaluate with CodeBLEU
python compare_codebleu.py

# Test functional correctness
python Test_Cases_Evaluation.py

# Auto-correction loop
python Auto-Correction_Loop.py
```

### Running Prompting Strategies (Experiment 2)

```bash
cd Experiment3

# Generate deobfuscated code with different strategies
python generate_deobfuscated_Code_Prompting_Strategy.py

# Compute readability scores
python Readability_Computation.py
```

**Note**: The `rsm.jar` file (83MB) for readability computation has been removed due to GitHub file size limits. Download it separately from the [JSEP 2018 paper authors](https://github.com/sealuzh/tools/tree/master/Readability).

### Running Contrastive Fine-Tuning (Experiment 3)

```bash
cd Experiment4-Contrastive-Fine-tuning

# API-based contrastive fine-tuning
python Contrastive_Fine-tuning.py

# Open-source model contrastive fine-tuning
python Open-source-Model_Contrastive_FT.py
```


## 📄 License

This project is for research purposes. Please ensure compliance with OpenAI's terms of service and relevant academic guidelines. See [LICENSE](LICENSE) for details.

## 👥 Authors

See the paper for author information.

---

**Note**: This repository represents a comprehensive study in bidirectional code transformation using LLMs. The findings reveal critical limitations in current fine-tuning paradigms, with implications for high-stakes software engineering deployment.
