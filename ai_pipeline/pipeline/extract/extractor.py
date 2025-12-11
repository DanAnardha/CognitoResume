# ai_pipeline/pipeline/extract/extractor.py

import json
import spacy
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Union
from spacy_layout import spaCyLayout

from .config import Config
from .data_processor import TextProcessor
from .file_utils import save_json, get_metadata_filename, save_metadata

class Extractor:
    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        self.config = Config(config_source)
        self.text_processor = TextProcessor(self.config.chunking_config)
        self.nlp_model = self.config.nlp_model

    def _extract_text_from_source(self, pdf_source: Union[str, bytes]) -> str:
        try:
            nlp_layout = spacy.load(self.nlp_model)
            layout = spaCyLayout(nlp_layout)
            if isinstance(pdf_source, bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_source)
                    tmp_pdf_path = tmp_pdf.name
                layout_doc = layout(tmp_pdf_path)
                Path(tmp_pdf_path).unlink()
            else:
                layout_doc = layout(pdf_source)
            return layout_doc._.markdown
        except OSError:
            print(f"Error: spaCy model '{self.nlp_model}' not found.")
            print("Run: python -m spacy download en_core_web_sm")
            raise
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def _generate_metadata(self, source_type: str, source_id: Any, total_chunks: int, output_chunks_file: str, status: str, processing_time: float, error_message: str = None) -> Dict[str, Any]:
        return {
            "source_identifier": source_id,
            "source_type": source_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extraction_details": {
                "status": status,
                "total_chunks": total_chunks,
                "output_chunks_file": output_chunks_file,
                "chunking_config_used": self.config.chunking_config,
                "nlp_model_used": self.nlp_model,
                "processing_time_seconds": round(processing_time, 2),
                "error_message": error_message
            },
            "auditing_notes": {
                "purpose": "To track the extraction process for quality control and debugging.",
                "what_is_tracked": [
                    "Source type (e.g., 'file_system', 'database').",
                    "Source identifier (e.g., file path, DB ID).",
                    "Extraction parameters (chunk size, overlap, NLP model).",
                    "Success status and number of chunks generated.",
                    "Output file location.",
                    "Processing time and error messages."
                ]
            }
        }

    def extract_from_source(self, pdf_source: Union[str, bytes], source_type: str, source_id: Any, output_dir: str = "ai_pipeline/data/output/extract", prefix: str = "chunks_") -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        
        if isinstance(source_id, str):
            source_name = Path(source_id).stem
        else:
            source_name = "extracted_from_bytes"
            
        output_filename = f"{prefix}{source_name}.json"
        output_filepath = Path(output_dir) / output_filename
        metadata_path = get_metadata_filename(str(source_name))

        try:
            print(f"Memproses sumber dengan ID: {source_id} (dari tipe: {source_type})")
            raw_markdown = self._extract_text_from_source(pdf_source)
            cleaned_markdown = self.text_processor.clean_markdown(raw_markdown)
            chunks = self.text_processor.chunk_text(cleaned_markdown)
            save_json(chunks, str(output_filepath))
            
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            metadata = self._generate_metadata(
                source_type=source_type,
                source_id=source_id,
                total_chunks=len(chunks),
                output_chunks_file=str(output_filepath),
                status="success",
                processing_time=processing_time
            )
            save_metadata(metadata, metadata_path)
            return {"chunks": chunks, "metadata": metadata}

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            print(f"Error occured when extracting: {e}")
            error_metadata = self._generate_metadata(
                source_type=source_type,
                source_id=source_id,
                total_chunks=0,
                output_chunks_file=str(output_filepath),
                status="failed",
                processing_time=processing_time,
                error_message=str(e)
            )
            save_metadata(error_metadata, metadata_path)
            return {"chunks": [], "metadata": error_metadata}

    def extract_from_file(self, input_path: str, output_dir: str = "ai_pipeline/data/output/extract", prefix: str = "chunks_") -> Dict[str, Any]:
        return self.extract_from_source(
            pdf_source=input_path,
            source_type="file_system",
            source_id=input_path,
            output_dir=output_dir,
            prefix=prefix
        )