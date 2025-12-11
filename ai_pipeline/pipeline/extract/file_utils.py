# ai_pipeline/pipeline/extract/file_utils.py

from ai_pipeline.pipeline.parse.file_utils import (
    save_json,
    get_metadata_filename,
    save_metadata
)

def get_metadata_filename(input_path: str, output_dir: str = "ai_pipeline/data/metadata/extract") -> str:
    from pathlib import Path
    input_path_obj = Path(input_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    filename = f"metadata_{input_path_obj.stem}.json"
    return str(output_dir_path / filename)