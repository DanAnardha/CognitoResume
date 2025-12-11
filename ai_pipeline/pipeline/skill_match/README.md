# AI-Powered Skill Matching Pipeline

This module provides a robust and configurable system to match candidate skills against job requirements. It uses a combination of **semantic similarity** (from sentence transformers) and **lexical similarity** (from fuzzy matching) to calculate a nuanced match score.

> **Note**: This module focuses on calculating similarity scores. It does **not** use Large Language Models (LLMs) to generate text. The LLMs are used in the separate `parse` and `evaluate` module, which this module consumes as input.

## Features

-   **Hybrid Similarity Scoring**: Combines semantic similarity (from sentence transformers) with lexical similarity (from fuzzy matching) for a more accurate and nuanced score.
-   **Configurable Scoring & Thresholds**: Fine-tune matching behavior (e.g., what constitutes a "strong" vs. "weak" match) and scoring weights for different skill categories.
-   **Data Normalization**: Pre-processes skills and job descriptions with configurable steps like lowercasing, punctuation removal, synonym application, and acronym expansion.
-   **Schema-Driven Output**: Validates and structures the output against a predefined JSON schema to ensure consistency.
-   **Comprehensive Metadata Tracking**: Automatically generates a detailed metadata file for every matching process, tracking status, configuration used, processing time, and results summary.
-   **Secure Configuration**: Sensitive data (like API keys or URLs) is automatically redacted in the metadata for security.
-   **Modular & Reusable Design**: The core logic is encapsulated in the `SkillMatcher` class, making it easy to integrate into other systems (e.g., orchestrated by a workflow manager or a task queue like Celery).

## How It Works: A Two-Stage Process

The AI Screener application typically works in two main stages:

1.  **Parsing Stage (`parse` module)**: An LLM (e.g., GPT-4) is used to extract structured data (like skills, experience, etc.) from an unstructured resume text.
2.  **Matching Stage (`skill_match` module)**: The structured skills from the first stage are then compared against the job description using this module to calculate a match score.

This module performs the second stage. It takes the list of skills from the parsed resume and compares them to the required and optional skills from a job description.

## Modular Structure

The module is divided into several components to separate concerns and improve maintainability:

-   `config.py`: Handles loading and merging of `skill_match_config.json` and `global_providers.json`.
-   `data_processor.py`: Contains the `TextNormalizer` class for cleaning and normalizing text based on configuration.
-   `skill_matcher.py`: The main orchestrator class (`SkillMatcher`) that coordinates the entire matching process.
-   `main.py`: The command-line interface (CLI) for running the pipeline.
-   `data/`: Directory for holding data files like `synonim.json` and `acronym.json`.
-   `skill_match_config.json`: The main configuration file for this module.

## Prerequisites

-   Python 3.8+
-   The `sentence-transformers` library. Install with:
    ```bash
    pip install sentence-transformers
    ```
-   The `json-repair` library. Install with:
    ```bash
    pip install json-repair
    ```
-   The `rapidfuzz` library. Install with:
    ```bash
    pip install rapidfuzz
    ```
-   Valid credentials or access for at least one configured LLM provider (e.g., `OPENAI_API_KEY`) in your environment. This is required by the `parse` module that provides the input to this module.

## Configuration

The module is configured via `config.json`. Key sections include:

-   `model_settings`: Specifies the sentence transformer model to use for semantic similarity (e.g., `all-mpnet-base-v2`).
-   `scoring_weights`: Defines the weights for different components of the final score (e.g., `required_skill`, `optional_skill`).
-   `skill_thresholds`: Sets the confidence thresholds for categorizing matches (`strong`, `weak`, `nice`).
-   `similarity_weights`: Balances the influence of semantic vs. lexical similarity.
-   `normalization`: A dictionary of booleans to control text preprocessing steps.
-   `synonym_file` & `acronym_file`: Paths to JSON files containing mappings for text normalization.

**Example `config.json`:**
```json
{
  "version": "1.0.0",
  "model_settings": {
    "model_name": "all-mpnet-base-v2"
  },
  "scoring_weights": {
    "required_skill": 0.30,
    "optional_skill": 0.20
  },
  "skill_thresholds": {
    "strong": 0.7,
    "weak": 0.5,
    "nice": 0.6
  },
  "similarity_weights": {
    "semantic": 0.8,
    "lexical": 0.2
  },
  "normalization": {
    "lowercase": true,
    "remove_punctuation": true,
    "strip_whitespace": true,
    "apply_synonyms": true,
    "apply_acronyms": true
  },
  "synonym_file": "ai_pipeline/pipeline/skill_match/data/synonim.json",
  "acronym_file": "ai_pipeline/pipeline/skill_match/data/acronym.json"
}
```

## Usage

### Basic Command

Run the pipeline from the project root directory. It is crucial to run it as a module.

```bash
python -m ai_pipeline.pipeline.skill_match.main --candidate-skills path/to/candidate.json --job-description path/to/job.json
```

### CLI Arguments

### CLI Arguments

| Argument | Short | Default | Description |
|---------|--------|----------|-------------|
| `--candidate-skills` | `-c` | (Required) | Path to the JSON file containing the list of candidate skills. |
| `--job-description` | `-j` | (Required) | Path to the JSON file containing job requirements. |
| `--output` | `-o` | Auto-generated | Path to the output JSON file. |
| `--config` | `-C` | `ai_pipeline/pipeline/skill_match/skill_match_config.json` | Path to the skill matching configuration file. |
| `--global-providers` | `-g` | `ai_pipeline/pipeline/config/global_providers.json` | Path to the global LLM providers config. |


### Example Usage

```bash
python -m ai_pipeline.pipeline.skill_match.main \
  --candidate-skills data/output/parse/candidate_101_parsed.json \
  --job-description data/input/job_description_55.json \
  --output data/output/skill_match/match_result.json
```

## Input & Output

### Input

1.  **Candidate Skills File (JSON)**: An array of strings, usually the `skills.items` from the parsed resume.
    ```json
    [
      "Python", "Django", "PostgreSQL", "Docker", "AWS", "TensorFlow"
    ]
    ```

2.  **Job Description File (JSON)**: An object with two keys: `required_skills` and `optional_skills`, both arrays of strings.
    ```json
    {
      "required_skills": ["Python", "Django or Flask", "SQL"],
      "optional_skills": ["AWS", "Docker", "React"]
    }
    ```

### Output

The pipeline generates two files for each run:

1.  **Matching Results (JSON)**: Contains the detailed matching results, including scores, lists of matches, and a summary.
    -   **Default Location**: `ai_pipeline/data/output/skill_match/[job_filename]_matching_results.json`

2.  **Metadata (JSON)**: Contains comprehensive audit information about the process.
    -   **Default Location**: `ai_pipeline/data/metadata/skill_match/metadata_[candidate_filename]_vs_[job_filename].json`

**Example Metadata Structure:**
```json
{
  "source_identifiers": {
    "candidate": { "type": "file_system", "id": "path/to/candidate.json" },
    "job": { "type": "file_system", "id": "path/to/job.json" }
  },
  "timestamp": "2023-11-15T11:00:00.123456+00:00",
  "matching_details": {
    "status": "success",
    "config_used": {
      "model_name": "all-mpnet-base-v2",
      "thresholds": { ... },
      "weights": { ... },
      "normalization": { ... },
      "data_files": { ... }
    },
    "processing_time_seconds": 2.15,
    "error_message": null
  },
  "matching_results_summary": {
    "overall_score": 0.8125,
    "required_match_score": 0.85,
    "optional_match_score": 0.75,
    "summary_counts": { ... }
  }
}
```