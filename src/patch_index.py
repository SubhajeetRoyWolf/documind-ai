"""
Patch Pinecone index with specific facts from resume
that may have been lost during chunking.
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from ingestion import ingest_text_directly

# Key facts that must be retrievable
facts = [
    "Subhajeet Roy works at Saks Global India in Bengaluru as a Data Science and AI Integration Contributor and QA Automation Analyst since 2022.",
    "The candidate has 3.5+ years of experience in data science and ML engineering.",
    "Current employer is Saks Global India located in Bengaluru.",
    "Saks Global India is where the candidate has worked from 2022 to present.",
]

print("Patching Pinecone index with key facts...")
for fact in facts:
    ingest_text_directly(fact, source="resume_patch.txt")

print("\n✅ Index patched! Key facts now searchable.")