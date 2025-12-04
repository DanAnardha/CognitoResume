# test_runner.py
import json
import pyperclip
from skill_matcher import SkillMatcher

# ===========================================
# Init matcher
# ===========================================

matcher = SkillMatcher()

# ===========================================
# Test Cases (A–H)
# ===========================================

tests = {
    "A": {
        "candidate": ["Email Marketing", "Copywriting", "Google Ads"],
        "required": ["Python", "SQL", "Machine Learning"],
        "optional": ["AWS", "MLOps"]
    },
    "B": {
        "candidate": ["A/B Testing", "Basic Statistics"],
        "required": ["Machine Learning", "Deep Learning", "Model Evaluation"],
        "optional": ["Big Data Technologies"]
    },
    "C": {
        "candidate": ["Git", "JavaScript", "SQL"],
        "required": ["SEO", "Content Writing", "Social Media Management"],
        "optional": ["A/B Testing"]
    },
    "D": {
        "candidate": ["Machine Learning", "Statistics"],
        "required": ["Figma", "UI Design", "UX Research"],
        "optional": ["Wireframing", "Prototyping"]
    },
    "E": {
        "candidate": ["A/B Testing", "Basic Statistics"],
        "required": ["Deep Learning", "TensorFlow", "PyTorch"],
        "optional": ["Machine Learning"]
    },
    "F": {
        "candidate": ["SQL", "APIs", "Node.js"],
        "required": ["Backend", "APIs", "SQL"],
        "optional": ["Docker"]
    },
    "G": {
        "candidate": ["Basic Data Analysis"],
        "required": ["Machine Learning", "Deep Learning", "Model Evaluation", "Data Wrangling"],
        "optional": ["Big Data Technologies"]
    },
    "H": {
        "candidate": ["Python", "SQL", "Statistics"],
        "required": ["Python", "SQL", "Statistics", "Machine Learning", "Deep Learning", "Data Wrangling"],
        "optional": ["AWS", "MLOps"]
    },
}

# ===========================================
# RUN ALL TESTS
# ===========================================

all_results = {}

print("\n========== RUNNING TEST CASES ==========\n")

for label, case in tests.items():
    print(f"Running test {label}...")

    res = matcher.match_skills(case["candidate"], case["required"], case["optional"])
    report = matcher.generate_report(case["candidate"], case["required"], case["optional"], res)

    all_results[label] = report

print("\n========== FINISHED ==========\n")

# ===========================================
# COPY OUTPUT TO CLIPBOARD
# ===========================================

json_output = json.dumps(all_results, indent=2)
pyperclip.copy(json_output)

print("✔ All test results have been copied to clipboard.")
print("✔ Paste anywhere (CTRL+V / CMD+V).")

# Print output
print("\n========== RESULT JSON ==========\n")
print(json_output)