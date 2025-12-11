# ai_pipeline/pipeline/parse/parser.py

from typing import Dict, Any, List, Union
from .config import Config
from .data_processor import ResumeProcessor
from ..llm_providers.manager import LLMManager
from .file_utils import load_json, save_json, save_metadata, get_metadata_filename, get_output_filename
import datetime

def _sanitize_provider_config_for_metadata(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    SENSITIVE_KEYS = {
        "api_key", "url", "model_path", "api_base" 
    }
    
    sanitized_config = {}
    for key, value in provider_config.items():
        if key in SENSITIVE_KEYS:
            sanitized_config[key] = "[REDACTED]"
        else:
            sanitized_config[key] = value
            
    return sanitized_config

def generate_metadata(
    parsed_data: Dict[str, Any], 
    source_type: str,
    source_id: Any,
    provider_used: str,
    config: Config,
    llm_manager: LLMManager,
    output_file: str,
    processing_time: float
) -> Dict[str, Any]:
    full_provider_config = config.get_provider_config(provider_used)
    safe_provider_config = _sanitize_provider_config_for_metadata(full_provider_config)
    
    metadata = {
        "source_identifier": {
            "type": source_type,
            "id": source_id
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "parsing_details": {
            "status": "success",
            "provider_used": provider_used,
            "provider_config": safe_provider_config,
            "pipeline_config": config.get_pipeline_config(provider_used),
            "output_file": output_file,
            "processing_time_seconds": round(processing_time, 2),
            "total_steps_executed": len(config.get_pipeline_config(provider_used).get("steps", [])),
            "error_message": None
        },
        "parsing_results": {
            "summary_confidence": parsed_data.get("summary", {}).get("confidence", 0),
            "sections": {}
        },
        "auditing_notes": {
            "purpose": "To track the parsing process for quality control and debugging.",
            "what_is_tracked": [
                "Source identifier (type and ID, e.g., 'file_system', 'database').",
                "Provider configuration used (sensitive data is redacted).",
                "Pipeline steps executed.",
                "Success status and processing time.",
                "Output file location.",
                "Confidence scores for each section.",
                "Number of items extracted per section."
            ]
        }
    }
    
    for section in ["education", "work_experience", "certifications", "projects", "skills"]:
        if section in parsed_data:
            section_data = parsed_data[section]
            metadata["parsing_results"]["sections"][section] = {
                "confidence": section_data.get("confidence", 0),
                "item_count": len(section_data.get("items", [])) if "items" in section_data else 0
            }
    
    return metadata

def parse_resume_data(
    input_chunks: List[str],
    config: Union[Config, None] = None,
    config_path: str = "ai_pipeline/pipeline/parse/config.json",
    global_providers_path: str = "ai_pipeline/pipeline/config/global_providers.json",
    source_type: str = "file_system",
    source_id: Any = "unknown_resume"
) -> Dict[str, Any]:
    start_time = datetime.datetime.now()
    
    try:
        if config is None:
            config = Config(parse_config=config_path, global_providers=global_providers_path)
        
        schema_path = "ai_pipeline/pipeline/parse/schema.json"
        schema = load_json(schema_path)
        if not schema:
            raise ValueError("Schema could not be loaded. Exiting.")

        providers_config = config.get_all_providers_config()
        llm_manager = LLMManager(providers_config)
        
        if not llm_manager.providers:
            raise ValueError("Could not initialize any LLM provider.")
            
        processor = ResumeProcessor(schema)

        final_result = processor.process_resume(
            llm_manager=llm_manager, 
            chunks=input_chunks, 
            config=config
        )
        
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        output_path = get_output_filename(source_id)
        
        provider_used = llm_manager.get_used_provider() or config.active_provider
        metadata = generate_metadata(
            parsed_data=final_result, 
            source_type=source_type,
            source_id=source_id,
            provider_used=provider_used,
            config=config,
            llm_manager=llm_manager,
            output_file=output_path,
            processing_time=processing_time
        )
        
        return {
            "result": final_result,
            "metadata": metadata
        }

    except Exception as e:
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"An error occurred in parsing pipeline: {e}")
        
        error_metadata = {
            "source_identifier": {
                "type": source_type,
                "id": source_id
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "parsing_details": {
                "status": "failed",
                "provider_used": None,
                "provider_config": None,
                "pipeline_config": None,
                "output_file": None,
                "processing_time_seconds": round(processing_time, 2),
                "total_steps_executed": 0,
                "error_message": str(e)
            },
            "parsing_results": {
                "summary_confidence": 0,
                "sections": {}
            },
            "auditing_notes": {
                "purpose": "To track the parsing process for quality control and debugging.",
                "what_is_tracked": [
                    "Source identifier (type and ID, e.g., 'file_system', 'database').",
                    "Provider configuration used (sensitive data is redacted).",
                    "Pipeline steps executed.",
                    "Success status and processing time.",
                    "Output file location.",
                    "Error messages for debugging."
                ]
            }
        }
        
        return {
            "result": {},
            "metadata": error_metadata
        }