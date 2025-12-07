# backend/prototypes/skill_match/test_skill_matcher.py

import json
import os
import re
from skill_matcher import EnhancedSkillMatcher

test_cases = [
    {
        "test_name": "Finance Specialist",
        "candidate_skills": [
            "Financial accounting", "Financial planning", "Financial reporting", "Financial analysis",
            "Account reconciliations", "Critical thinking", "General ledger", "Enterprise Resource Planning",
            "ERP software", "Defense Enterprise Accounting and Management System (DEAMS)", "Budget coordination",
            "Budget management", "Resource advising", "Defense Travel System (DTS)",
            "General Accounting and Finance System (GAFS/BQ)", "Government Purchase Card (GPC)",
            "Accounting operations", "Analysis of financial systems", "Testing scripts and patches",
            "System change requests", "Performance metrics", "Vendor pay", "Accounts payable management",
            "Supervisory accounting", "Staff supervision", "Training development", "Remote training",
            "Process improvement", "Workflow analysis", "Data analysis", "Louis II data retrieval software",
            "Reporting", "Managerial reports", "Reconciliation", "Budget oversight", "Audit support",
            "Compliance", "Federal Managers Financial Integrity Act (FMFIA)", "Foreign currency fluctuation reporting",
            "Customer service", "Stakeholder communication", "Accounting", "General Accounting",
            "Accounts Payable", "Program Management"
        ],
        "job_required_skills": [
            "Financial Accounting", "Financial Reporting", "General Ledger Management",
            "Enterprise Resource Planning Systems", "Team Leadership", "Process Improvement",
            "Budget Management", "Regulatory Compliance", "Department of Defense Financial Systems"
        ],
        "job_optional_skills": [
            "DEAMS", "GAFS/BQ", "Accounts Payable Management", "Training Development",
            "Audit Support", "Stakeholder Communication", "System Implementation", "Change Management"
        ]
    },
    {
        "test_name": "Software Engineer (High Match)",
        "candidate_skills": [
            "Python Programming", "Software Development", "REST API Design", "Agile Methodologies",
            "Version Control with Git", "Docker and Kubernetes", "AWS Cloud Services", "CI/CD Pipelines",
            "Database Management (SQL)", "Unit Testing", "System Design", "Problem Solving"
        ],
        "job_required_skills": [
            "Python", "Software Development", "API Design", "Agile", "Git"
        ],
        "job_optional_skills": [
            "Cloud Computing", "CI/CD", "Docker"
        ]
    },
    {
        "test_name": "Data Scientist (Partial Match with OR)",
        "candidate_skills": [
            "Python Programming", "Data Analysis with Pandas", "Machine Learning Models (Scikit-learn)",
            "Data Visualization with Matplotlib", "SQL Database Queries", "Statistical Analysis",
            "Deep Learning with TensorFlow", "Natural Language Processing (NLP)", "AWS S3 and EC2"
        ],
        "job_required_skills": [
            "Proficiency in Python or R", "Experience with Machine Learning frameworks",
            "Data Wrangling and Analysis", "Statistical Modeling"
        ],
        "job_optional_skills": [
            "Deep Learning", "Big Data Technologies (Spark or Hadoop)", "Cloud Platform Experience (AWS or Azure)"
        ]
    },
    {
        "test_name": "Marketing Manager (Low Match)",
        "candidate_skills": [
            "Content Creation", "Social Media Strategy", "SEO and SEM", "Campaign Management",
            "Market Research", "Brand Development", "Graphic Design (Adobe Creative Suite)",
            "Copywriting", "Public Relations", "Email Marketing"
        ],
        "job_required_skills": [
            "Financial Planning", "Budget Analysis", "Risk Management", "Compliance Reporting"
        ],
        "job_optional_skills": [
            "Data Science", "Statistical Modeling", "Programming in Python"
        ]
    },
    {
        "test_name": "IT Support (Testing Acronyms)",
        "candidate_skills": [
            "Troubleshooting hardware and software issues",
            "Installing and configuring operating systems (Windows, macOS)",
            "Network setup and maintenance (TCP/IP, DNS)",
            "Providing technical support via phone and email",
            "Managing user accounts in Active Directory",
            "Using ticketing systems like Zendesk",
            "Knowledge of gafs/bq and dts systems",
            "Remote desktop assistance",
            "Hardware inventory management",
            "Creating technical documentation"
        ],
        "job_required_skills": [
            "Technical Support", "Operating Systems", "Networking", "Customer Service"
        ],
        "job_optional_skills": [
            "Active Directory", "Zendesk", "DEAMS", "DTS"
        ]
    }
]

def run_all_tests():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing EnhancedSkillMatcher...")
    matcher = EnhancedSkillMatcher()
    print("Matcher initialized. Starting test cases...\n")

    for i, case in enumerate(test_cases):
        print(f"--- Running Test Case {i+1}/{len(test_cases)}: {case['test_name']} ---")
        
        result = matcher.match_skills(
            candidate_skills=case["candidate_skills"],
            job_required=case["job_required_skills"],
            job_optional=case["job_optional_skills"]
        )
        
        test_slug = case["test_name"].lower()
        test_slug = re.sub(r'[\s/]+', '_', test_slug)
        test_slug = re.sub(r'[()]', '', test_slug)
        output_filename = os.path.join(output_dir, f"test_{i+1:02d}_{test_slug}_result.json")
    
        with open(output_filename, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {output_filename}")
        print(f"Final Score: {result['score']}\n")


if __name__ == "__main__":
    run_all_tests()