import json
import re
import requests
from typing import Any, Dict, List, Union, Set, Tuple
from collections.abc import Mapping, Sequence

# External libraries
try:
    import spacy
    from spacy_layout import spaCyLayout
    from json_repair import repair_json
except ImportError as e:
    print(f"Error: Required library not found. {e}")
    print("Please ensure you have installed: spacy, spacy-layout, json-repair, requests")
    exit()

# --- Default Schema ---
# This schema is used as a template and for validation.
DEFAULT_SCHEMA = {
    "summary": {"value": "", "confidence": 0.00},
    "education": {"items": [{"degree": "", "graduation_year": "", "institution": "", "gpa": ""}], "confidence": 0.00},
    "work_experience": {"items": [{"role": "", "company": "", "description": "", "years": ""}], "confidence": 0.00},
    "projects": {"items": [{"title": "", "role": "", "year": "", "description": "", "tech_stack": []}], "confidence": 0.00},
    "certifications": {"items": [{"name": "", "issuer": "", "year": ""}], "confidence": 0.00},
    "skills": {"items": [], "confidence": 0.00}
}

# --- Utility Functions ---

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' is invalid.")
        exit()

def save_to_json(data: Dict[str, Any], filepath: str):
    """Saves data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n--- Final result saved to '{filepath}' ---")

def clean_markdown(text: str) -> str:
    """Cleans raw markdown text extracted from a PDF."""
    text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    text = text.replace("�", " ").replace("·", "- ").replace("•", "- ").replace("○", "- ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text).replace("\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Splits text into smaller, overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def call_llm_api(api_url: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, timeout: int) -> str:
    """Reusable function to call the LLM API with streaming."""
    payload = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": temperature}
    response_text = ""
    try:
        with requests.post(f"{api_url}/chat/stream", json=payload, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.decode('utf-8').startswith("data: "):
                    json_data = line.decode('utf-8')[len("data: "):]
                    data = json.loads(json_data)
                    content = data.get('content', '')
                    if content:
                        response_text += content
                        print(content, end="", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"\nError contacting API: {e}")
        return ""
    return response_text

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parses and repairs the JSON response from the LLM."""
    if not response_text:
        return json.loads(json.dumps(DEFAULT_SCHEMA))
    
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    raw_json = match.group(0) if match else cleaned

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(raw_json)
            return json.loads(repaired)
        except Exception:
            print("Failed to repair JSON. Returning default schema.")
            return json.loads(json.dumps(DEFAULT_SCHEMA))

# --- Local JSON Normalization Functions (from user's code) ---

def normalize_json_preserve_structure(data: Any, 
                                     remove_empty: bool = True,
                                     remove_duplicates: bool = True,
                                     case_sensitive_duplicates: bool = True,
                                     preserve_order: bool = True,
                                     deep_copy: bool = True) -> Any:
    """Normalize JSON data while preserving structure of partially filled objects."""
    if deep_copy:
        data = json.loads(json.dumps(data))
    
    if isinstance(data, Mapping):
        return _normalize_dict_preserve(data, remove_empty, remove_duplicates, 
                                       case_sensitive_duplicates, preserve_order)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return _normalize_list_preserve(data, remove_empty, remove_duplicates,
                                      case_sensitive_duplicates, preserve_order)
    else:
        return _normalize_primitive(data, remove_empty)

def _normalize_dict_preserve(data: Dict[str, Any],
                           remove_empty: bool,
                           remove_duplicates: bool,
                           case_sensitive_duplicates: bool,
                           preserve_order: bool) -> Dict[str, Any]:
    """Normalize a dictionary while preserving structure of partially filled objects."""
    result = {}
    for key, value in data.items():
        normalized_value = normalize_json_preserve_structure(
            value, remove_empty, remove_duplicates,
            case_sensitive_duplicates, preserve_order, deep_copy=False
        )
        result[key] = normalized_value
    if remove_empty and _is_completely_empty(result):
        return {}
    return result

def _normalize_list_preserve(data: List[Any],
                           remove_empty: bool,
                           remove_duplicates: bool,
                           case_sensitive_duplicates: bool,
                           preserve_order: bool) -> List[Any]:
    """Normalize a list while preserving structure of partially filled objects."""
    result = []
    seen = set()
    for item in data:
        normalized_item = normalize_json_preserve_structure(
            item, remove_empty, remove_duplicates,
            case_sensitive_duplicates, preserve_order, deep_copy=False
        )
        if remove_empty and _is_completely_empty(normalized_item):
            continue
        if remove_duplicates:
            item_key = _make_hashable(normalized_item, case_sensitive_duplicates)
            if item_key in seen:
                continue
            seen.add(item_key)
        result.append(normalized_item)
    return result

def _normalize_primitive(data: Any, remove_empty: bool) -> Any:
    """Normalize a primitive value (string, number, boolean, null)."""
    if isinstance(data, str):
        cleaned = data.strip()
        return cleaned
    return data

def _is_completely_empty(value: Any) -> bool:
    """Check if a value is completely empty (all nested values are empty)."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    if isinstance(value, dict):
        return all(_is_completely_empty(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_completely_empty(item) for item in value)
    return False

def _make_hashable(value: Any, case_sensitive: bool = True) -> Union[Tuple, str, int, float, bool, None]:
    """Convert a value to a hashable representation for duplicate detection."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _make_hashable(v, case_sensitive)) for k, v in value.items()))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_make_hashable(item, case_sensitive) for item in value)
    elif isinstance(value, str):
        return value if case_sensitive else value.lower()
    return value

def local_structural_cleaning(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Specialized function for initial cleaning of resume data while preserving structure."""
    # First pass: normalize with structure preservation
    normalized = normalize_json_preserve_structure(resume_data)
    
    # Second pass: apply resume-specific cleaning rules
    # Handle skills section with items and confidence
    if 'skills' in normalized and isinstance(normalized['skills'], dict) and 'items' in normalized['skills']:
        skills = [skill.strip() for skill in normalized['skills']['items'] if isinstance(skill, str) and skill.strip()]
        # Remove duplicates while preserving order
        unique_skills = []
        seen_skills = set()
        for skill in skills:
            if skill.lower() not in seen_skills:
                seen_skills.add(skill.lower())
                unique_skills.append(skill)
        normalized['skills']['items'] = unique_skills
    
    return normalized

# --- Main Processing Functions ---

def merge_results(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merges data from a new chunk into the main results dictionary."""
    # Update summary if confidence is higher
    if new.get("summary", {}).get("confidence", 0) > base.get("summary", {}).get("confidence", 0):
        base["summary"] = new.get("summary", base["summary"])
        
    # Merge lists from sections with items. Redundancy and empty items will be handled later.
    for key in ["education", "work_experience", "certifications", "projects"]:
        if key in new and "items" in new[key] and isinstance(new[key]["items"], list):
            if new[key]["items"]:
                base[key]["items"].extend(new[key]["items"])
                if new[key].get("confidence", 0) > base[key].get("confidence", 0):
                    base[key]["confidence"] = new[key]["confidence"]

    # Skills: merge all items. Deduplication will be handled later.
    if "skills" in new and "items" in new["skills"] and isinstance(new["skills"]["items"], list):
        base["skills"]["items"].extend(new["skills"]["items"])
        if new["skills"].get("confidence", 0) > base["skills"].get("confidence", 0):
            base["skills"]["confidence"] = new["skills"]["confidence"]
    return base

def extract_text_from_pdf(pdf_path: str, spacy_model_name: str) -> str:
    """Extracts and cleans text from a PDF file using spaCy."""
    try:
        nlp_layout = spacy.load(spacy_model_name)
        layout = spaCyLayout(nlp_layout)
        layout_doc = layout(pdf_path)
        raw_markdown = layout_doc._.markdown
        cleaned_markdown = clean_markdown(raw_markdown)
        print("--- Markdown text extracted and cleaned ---")
        return cleaned_markdown
    except OSError:
        print(f"Error: spaCy model '{spacy_model_name}' not found.")
        print("Run: python -m spacy download en_core_web_sm")
        exit()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        exit()

def stage1_extraction(api_config: Dict, prompts_config: Dict, chunks: List[str]) -> Dict[str, Any]:
    """Stage 1: Extracts raw data from each text chunk using the LLM."""
    extraction_rules = """
    3. Work Experience must follow strict definitions and include ONLY real employment.
       - "role": the official job title.
       - "company": the name of the employer.
       - "description": a summary of responsibilities and achievements.
       - "years": the period of employment (e.g., "2020-2023").
    4. Projects must follow strict definitions and include ONLY actual projects.
       - "title": the official name of the project.
       - "role": the function held within the project.
       - "year": the year the project was active or completed.
       - "description": a brief summary of the project's purpose.
       - "tech_stack": a list of individual technologies used.
    5. Education must follow strict definitions and include ONLY formal degrees.
       - "degree": the full name of the degree obtained.
       - "graduation_year": the year of graduation.
       - "institution": the name of the university or school.
       - "gpa": the Grade Point Average, if mentioned.
    6. Certifications must follow strict definitions:
       - "name": the certification title only.
       - "issuer": must be an organization/company/school. NEVER a program name, course name, role, or year.
       - "year": must be a 4-digit year ONLY.
    7. Skills must ONLY contain a list of strings. Extract EVERY skill, sub-skill, method, library, framework, algorithm, and tool INDIVIDUALLY.
    """
    
    schema_string = json.dumps(DEFAULT_SCHEMA, indent=2)
    base_prompt_template = """
    You are a resume text classifier. Extract information from the resume and return valid JSON.
    Follow EXACTLY the schema below:
    {schema}

    STRICT RULES:
    1. Classify text based on meaning, NOT position.
    2. If a section is missing, return an empty list/string AND set confidence to 0.00.
    {rules}
    8. Do not introduce objects inside the skills list. Only strings are allowed.
    9. Never invent or infer missing information.
    10. Use the schema without modification.

    CONFIDENCE RULES (non-categorical):
    - Confidence must be a floating-point number between 0.00 and 1.00 (two decimals).

    Now extract the content strictly using the provided schema from this CV chunk:
    {chunk}

    Output only VALID JSON. No text before or after.
    """

    merged_result = json.loads(json.dumps(DEFAULT_SCHEMA))  # Deep copy

    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
        
        user_prompt = base_prompt_template.format(
            schema=schema_string,
            rules=extraction_rules.strip(),
            chunk=chunk
        )
        
        messages = [
            {"role": "system", "content": prompts_config["system_stage1"]},
            {"role": "user", "content": user_prompt},
        ]
        
        response_text = call_llm_api(
            api_config["url"], messages, api_config["max_tokens"], 
            api_config["temperature"], api_config["timeout"]
        )
        
        partial_result = parse_llm_response(response_text)
        merged_result = merge_results(merged_result, partial_result)
        
    return merged_result

def stage2_post_processing(api_config: Dict, prompts_config: Dict, initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: Cleans and normalizes the extracted data using the LLM."""
    print("\n--- Starting Stage 2: Post-Processing ---")
    
    schema_string = json.dumps(DEFAULT_SCHEMA, indent=2)
    input_json_string = json.dumps(initial_data, indent=2)

    post_processing_prompt_template = """
    You are a post-processing engine. Your ONLY job is to normalize, clean, validate, and restructure the parsed resume JSON **without adding, inferring, or inventing anything**.
    You do NOT re-extract from resume text. Input = parsed JSON only.

    GLOBAL RULES:
    - NEVER add new content.
    - NEVER infer missing data.
    - Keep EXACTLY the same schema, structure, and keys as DEFAULT_SCHEMA.
    - Preserve all values unless normalization is explicitly allowed.
    - Normalize or remove fields that the data doesn't seem right (e.g. year field filled with "US").

    ALLOWED NORMALIZATION ONLY:
    1. Remove bullet characters, emojis, decorative symbols.
    2. Trim whitespace, fix spacing, remove duplicate punctuation.
    3. Capitalization fixes: Title Case for titles, Proper Nouns for institutions/companies.
    4. Normalize known variants: "scikit learn" → "scikit-learn", "tensorflow" → "TensorFlow", "numpy" → "NumPy". (Do NOT add new variants. Only normalize what already exists.)

    SECTION RULES:
    ### Skills: Split on commas, remove adjectives, dedupe.
    ### Work Experience: Remove exact duplicates, preserve incomplete entries.
    ### Education: Normalize degree/institution capitalization.
    ### Certifications: DO NOT infer issuers/years. Only reformat/clean text.
    ### Projects: Keep tech_stack items as-is except formatting.

    CONFIDENCE: Must be a floating-point number between 0.00 and 1.00 (two decimals).
    DATES: Normalize ONLY if original value clearly indicates valid components to "YYYY-MM-DD", "YYYY-MM", OR "YYYY". DO NOT INVENT.

    OUTPUT: VALID JSON ONLY. No explanation.

    DEFAULT_SCHEMA:
    {schema}

    INPUT_JSON:
    {input_json}
    """
    
    user_prompt = post_processing_prompt_template.format(
        schema=schema_string,
        input_json=input_json_string
    )
    
    messages = [
        {"role": "system", "content": prompts_config["system_stage2"]},
        {"role": "user", "content": user_prompt},
    ]
    
    response_text = call_llm_api(
        api_config["url"], messages, api_config["max_tokens"], 
        api_config["temperature"], api_config["timeout"]
    )
    
    final_result = parse_llm_response(response_text)
    return final_result

# --- Final Validation and Cleaning Function ---

def final_validation_and_cleaning(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    A final deterministic step to ensure the data perfectly matches the schema.
    This acts as a safety net after the LLM's semantic cleaning.
    """
    print("\n--- Starting Final Validation and Structural Cleaning ---")
    # Create a deep copy of the schema to ensure all keys exist
    final_data = json.loads(json.dumps(schema))

    # Helper to validate and round confidence scores
    def validate_confidence(conf_val: Any) -> float:
        try:
            return round(float(conf_val), 2)
        except (ValueError, TypeError):
            return 0.00

    # Process summary
    if "summary" in data and isinstance(data["summary"], dict):
        final_data["summary"]["value"] = data["summary"].get("value", "")
        final_data["summary"]["confidence"] = validate_confidence(data["summary"].get("confidence"))

    # Process sections with lists of items
    for key in ["education", "work_experience", "certifications", "projects"]:
        if key in data and isinstance(data[key], dict) and "items" in data[key]:
            # Keep only non-empty items from the LLM's output
            non_empty_items = [
                item for item in data[key]["items"] 
                if not _is_completely_empty(item)
            ]
            final_data[key]["items"] = non_empty_items
            final_data[key]["confidence"] = validate_confidence(data[key].get("confidence"))

    # Process skills
    if "skills" in data and isinstance(data["skills"], dict):
        if isinstance(data["skills"].get("items"), list):
            # Ensure all items are strings and perform final deduplication
            skills_list = [str(skill).strip() for skill in data["skills"]["items"] if str(skill).strip()]
            unique_skills = []
            seen_skills = set()
            for skill in skills_list:
                if skill.lower() not in seen_skills:
                    seen_skills.add(skill.lower())
                    unique_skills.append(skill)
            final_data["skills"]["items"] = unique_skills
        final_data["skills"]["confidence"] = validate_confidence(data["skills"].get("confidence"))
        
    return final_data

def main():
    """Main function to orchestrate the entire resume processing pipeline."""
    config = load_config()
    
    # Stage 0: Extract and clean text from PDF
    cleaned_markdown = extract_text_from_pdf(config["pdf_path"], config["spacy_model"])
    chunks = chunk_text(cleaned_markdown, config["chunking"]["size"], config["chunking"]["overlap"])
    print(f"Text split into {len(chunks)} chunks.")
    
    # Stage 1: Extract information from each chunk
    initial_extraction_result = stage1_extraction(config["api"], config["prompts"], chunks)
    print("\n--- Stage 1 Extraction Result ---")
    print(json.dumps(initial_extraction_result, indent=2))
    
    # Stage 1.5: Local cleaning and normalization
    print("\n--- Starting Local Data Cleaning ---")
    cleaned_resume_data = local_structural_cleaning(initial_extraction_result)
    print("--- Local Data Cleaning Complete ---")

    # Stage 2: Post-processing and final cleaning with LLM
    llm_processed_result = stage2_post_processing(config["api"], config["prompts"], cleaned_resume_data)
    
    # Stage 3: Final validation and structural cleaning with Python
    truly_final_result = final_validation_and_cleaning(llm_processed_result, DEFAULT_SCHEMA)
    print("\n--- Final Validation and Cleaning Complete ---")
    print(json.dumps(truly_final_result, indent=2))

    # Save final result to JSON file
    save_to_json(truly_final_result, config["output_filename"])


if __name__ == "__main__":
    main()