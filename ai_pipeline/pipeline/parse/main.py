# ai_pipeline/pipeline/parse/main.py

import argparse
from pathlib import Path
import os
from dotenv import load_dotenv

dotenv_path = Path('../../.env') 
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

from .parser import parse_resume_data
from .file_utils import load_json, save_json, get_output_filename, save_metadata, get_metadata_filename

def main():
    parser = argparse.ArgumentParser(description='Parse resume from chunks.')
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True, 
        help='Path to the chunks JSON file.'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='',
        help='Path to the output JSON file.'
    )
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default='ai_pipeline/pipeline/parse/config.json',
        help='Path to the parse configuration file.'
    )
    parser.add_argument(
        '--global-providers', '-g', 
        type=str, 
        default='ai_pipeline/pipeline/config/global_providers.json',
        help='Path to the global providers configuration file.'
    )
    
    args = parser.parse_args()
    
    try:
        chunks = load_json(args.input)
        if not chunks:
            print("Exiting due to missing or empty input chunks.")
            return
        parsing_result = parse_resume_data(
            input_chunks=chunks,
            config_path=args.config,
            global_providers_path=args.global_providers,
            source_type="file_system",
            source_id=args.input
        )
        
        final_parsed_resume = parsing_result.get("result", {})
        metadata = parsing_result.get("metadata", {})
        if final_parsed_resume:
            if args.output:
                output_path = args.output
            else:
                output_path = get_output_filename(args.input)
            save_json(final_parsed_resume, output_path)
            if "parsing_details" in metadata:
                metadata["parsing_details"]["output_file"] = output_path
        else:
            print("Parsing failed or returned an empty result.")
            
        if metadata:
            metadata_path = get_metadata_filename(args.input)
            save_metadata(metadata, metadata_path)
        else:
            print("No metadata to save.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()