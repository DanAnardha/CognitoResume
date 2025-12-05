import json
import re
import requests
from typing import Dict, Any, List

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

def is_item_empty(item: Dict[str, Any]) -> bool:
    """Helper to check if a data item (e.g., a job entry) is effectively empty."""
    for value in item.values():
        if isinstance(value, str) and value.strip():
            return False  # Found a non-empty string
        if isinstance(value, list) and value:
            return False  # Found a non-empty list
    return True  # All values are empty

def merge_results(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merges data from a new chunk into the main results dictionary."""
    # Update summary if confidence is higher
    if new.get("summary", {}).get("confidence", 0) > base.get("summary", {}).get("confidence", 0):
        base["summary"] = new.get("summary", base["summary"])
        
    # Merge lists from sections with items, filtering out empty ones
    for key in ["education", "work_experience", "certifications", "projects"]:
        if key in new and "items" in new[key] and isinstance(new[key]["items"], list):
            # Filter out empty items from the new chunk before merging
            non_empty_items = [item for item in new[key]["items"] if not is_item_empty(item)]
            if non_empty_items:
                base[key]["items"].extend(non_empty_items)
                if new[key].get("confidence", 0) > base[key].get("confidence", 0):
                    base[key]["confidence"] = new[key]["confidence"]

    # Skills: dedupe and merge
    if "skills" in new and "items" in new["skills"] and isinstance(new["skills"]["items"], list):
        combined_skills = list(set(base["skills"]["items"] + new["skills"]["items"]))
        base["skills"]["items"] = combined_skills
        if new["skills"].get("confidence", 0) > base["skills"].get("confidence", 0):
            base["skills"]["confidence"] = new["skills"]["confidence"]
    return base

# --- Main Processing Functions ---

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
    
    # Stage 2: Post-processing and final cleaning
    final_result = stage2_post_processing(config["api"], config["prompts"], initial_extraction_result)
    
    # Save final result to JSON file
    save_to_json(final_result, config["output_filename"])


if __name__ == "__main__":
    main()