# PDF Text Extraction Pipeline

This module is responsible for extracting text from PDF files, cleaning it, and splitting it into smaller chunks for further processing.

## Features

-   **Text Extraction**: Uses `spaCy` and `spacy-layout` for high-precision text extraction from PDF files.
-   **Text Cleaning**: Cleans markdown artifacts and unwanted characters from the extracted text.
-   **Text Chunking**: Splits long text into configurable chunks based on size and overlap.
-   **Metadata Tracking**: Automatically generates a metadata file to track process status, configuration used, processing time, and output location.
-   **Modular Structure**: Designed with a modular architecture for easier maintenance, testing, and development.

## Modular Structure

The module is divided into several components to separate concerns:

-   `config.py`: Handles loading and access to the `config.json` file.
-   `data_processor.py`: Contains the logic for text cleaning and chunking.
-   `extractor.py`: The main orchestrator that coordinates the entire extraction process.
-   `file_utils.py`: Contains utility functions for file operations (saving chunks and metadata).
-   `main.py`: The command-line interface (CLI) for running the pipeline.
-   `config.json`: The configuration file for the NLP model and chunking parameters.

## Prerequisites

-   Python 3.8+
-   The English `spaCy` model. Install it by running:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

This pipeline is run via the command-line (CLI). **It is crucial to always run it as a module from the project root directory** to avoid `ModuleNotFoundError`.

### Basic Command

Navigate to your project's root directory in the terminal and run:

```bash
python -m ai_pipeline.pipeline.extract.main --input PATH/TO/YOUR/PDF/FILE.pdf
```

### CLI Arguments

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | (Required) | Path to the input PDF file. |
| `--output-dir` | `-o` | `ai_pipeline/data/output/extract` | Directory to save the output chunks file. |
| `--config` | `-c` | `ai_pipeline/pipeline/extract/config.json` | Path to the configuration file. |
| `--prefix` | `-p` | `chunks_` | Prefix for the output filename. |

### Usage Example

```bash
python -m ai_pipeline.pipeline.extract.main \
  --input data/raw/resume_danuar.pdf \
  --output-dir data/output/extract \
  --prefix chunks_
```

## Input & Output

### Input

-   **Single File**: One PDF file to be processed.

### Output

The pipeline generates two files for each run:

1.  **Chunks File (JSON)**:
    -   Contains an array of strings, where each string is a text chunk.
    -   Saved in the specified output directory (default: `ai_pipeline/data/output/extract/`).
    -   Filename follows the format: `[prefix][original_filename].json`.

2.  **Metadata File (JSON)**:
    -   Contains audit information about the extraction process, such as:
        -   Source file and timestamp.
        -   Status (`success` or `failed`).
        -   Configuration used (spaCy model, chunk size).
        -   Number of chunks generated and processing time.
        -   Error message (if the process failed).
    -   Saved in the `ai_pipeline/data/metadata/extract/` directory.
    -   Filename follows the format: `metadata_[original_filename].json`.

### Example Execution

After running the example command above, you will see output in your terminal:

```
Processing file from path: data/raw/resume_danuar.pdf
--- Text markdown successfully extracted ---
Text split into 2 chunks.
--- Text chunks saved to 'ai_pipeline/data/output/extract/chunks_resume_danuar.json' ---
--- Extraction metadata saved to 'ai_pipeline/data/metadata/extract/metadata_resume_danuar.json' ---

--- Extraction Process Successful ---
Chunks file saved at: ai_pipeline/data/output/extract/chunks_resume_danuar.json
Total chunks created: 2
Processing time: 3.45 seconds
Metadata saved at: ai_pipeline/data/metadata/extract/metadata_resume_danuar.json
```

And two files will be created:
-   `ai_pipeline/data/output/extract/chunks_resume_danuar.json`
-   `ai_pipeline/data/metadata/extract/metadata_resume_danuar.json`

