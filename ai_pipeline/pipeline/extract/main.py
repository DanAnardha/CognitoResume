# ai_pipeline/pipeline/extract/main.py

import argparse
from pathlib import Path
from .extractor import Extractor
from .file_utils import get_metadata_filename

def main():
    parser = argparse.ArgumentParser(description='Extract text from a PDF file and split it into chunks.')
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input PDF file (relative to the project root)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='ai_pipeline/data/output/extract',
        help='Output directory (relative to the project root)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='ai_pipeline/pipeline/extract/config.json',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default='chunks_',
        help='Prefix for output file names'
    )
    
    args = parser.parse_args()

    try:
        extractor = Extractor(args.config)
    except Exception as e:
        print(f"Failed to initialize Extractor: {e}")
        return

    result = extractor.extract_from_file(
        input_path=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix
    )
    
    metadata = result.get("metadata", {})
    details = metadata.get("extraction_details", {})

    if details.get("status") == "success":
        print("\n--- Extraction Successful ---")
        print(f"Chunks saved at: {details.get('output_chunks_file')}")
        print(f"Total chunks: {details.get('total_chunks')}")
        print(f"Processing time: {details.get('processing_time_seconds')} seconds")
    else:
        print("\n--- Extraction Failed ---")
        print(f"Error: {details.get('error_message')}")

    print(f"Metadata saved to: {get_metadata_filename(args.input)}")

if __name__ == "__main__":
    main()
