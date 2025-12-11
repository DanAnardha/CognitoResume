# ai_pipeline/pipeline/skill_match/main.py

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from .skill_matcher import SkillMatcher
from ai_pipeline.pipeline.parse.file_utils import save_json, save_metadata

def load_json(filepath: str) -> Any:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'")
        raise

def get_output_filename(input_path: str, output_dir: str = "ai_pipeline/data/output/skill_match", suffix: str = "_matching_results") -> str:
    input_path_obj = Path(input_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    filename = f"{input_path_obj.stem}{suffix}.json"
    return str(output_dir_path / filename)

def get_metadata_filename(candidate_path: str, job_path: str, output_dir: str = "ai_pipeline/data/metadata/skill_match") -> str:
    candidate_name = Path(candidate_path).stem
    job_name = Path(job_path).stem
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    filename = f"metadata_{candidate_name}_vs_{job_name}.json"
    return str(output_dir_path / filename)

def main():
    parser = argparse.ArgumentParser(
        description='Match candidate skills against a job description using semantic and lexical similarity.'
    )
    parser.add_argument(
        '--candidate-skills', '-cands',
        type=str,
        required=True,
        help='Path to the JSON file containing list of candidate skills.'
    )
    parser.add_argument(
        '--job-description', '-job',
        type=str,
        required=True,
        help='Path to the JSON file containing job description (required and optional skills).'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='',
        help='Path to the output JSON file for the matching results.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='ai_pipeline/pipeline/skill_match/config.json',
        help='Path to the skill matching configuration file.'
    )
    
    args = parser.parse_args()
    
    try:
        candidate_skills = load_json(args.candidate_skills)
        if not isinstance(candidate_skills, list):
            raise ValueError("Candidate skills file must contain a JSON array of strings.")

        job_description = load_json(args.job_description)
        if not isinstance(job_description, dict) or 'required_skills' not in job_description:
            raise ValueError("Job description file must be a JSON object with a 'required_skills' key.")
        
        required_skills = job_description.get('required_skills', [])
        optional_skills = job_description.get('optional_skills', [])

        matcher = SkillMatcher(config_source=args.config)
        matching_output = matcher.match_skills(
            candidate_skills=candidate_skills,
            job_required=required_skills,
            job_optional=optional_skills,
            candidate_source_type="file_system",
            candidate_id=args.candidate_skills,
            job_source_type="file_system",
            job_id=args.job_description
        )
        
        results = matching_output.get("results", {})
        metadata = matching_output.get("metadata", {})

        if args.output:
            output_path = args.output
        else:
            output_path = get_output_filename(args.job_description)
        
        save_json(results, output_path)
        
        metadata_path = get_metadata_filename(args.candidate_skills, args.job_description)
        save_metadata(metadata, metadata_path)

        if metadata.get("matching_details", {}).get("status") == "success":
            print("\n--- Matching Process Completed ---")
            summary = results.get("summary", {})
            print(f"Required Skills - Strong Match: {summary.get('req_strong', 0)}, Weak Match: {summary.get('req_weak', 0)}, Missing: {summary.get('req_missing', 0)}")
            print(f"Optional Skills - Relevant: {summary.get('nice_relevant', 0)}")
            print(f"Overall Skill Match Score: {results.get('score', 0):.4f}")
            print(f"\nFull results saved to: '{output_path}'")
        else:
            print("\n--- Matching Process Failed ---")
            print(f"See metadata file for details: '{metadata_path}'")
            
        print(f"Metadata saved to: '{metadata_path}'")

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"\n--- Aborted: Invalid Input or File Not Found ---")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Could not complete the skill matching process. Details: {e}")

if __name__ == "__main__":
    main()