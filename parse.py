from langchain.schema import Document
import re
import json

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from dotenv import load_dotenv
load_dotenv()

# parse PDF
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    input_files=["cases.pdf"], file_extractor=file_extractor
).load_data()

# print(documents)

full_text = "\n".join([doc.get_content() for doc in documents])
# print(full_text)

# split cases according to ALL CAPS 
pattern = r"(?=^[A-Z0-9 :,#]+$)"
cases_list = re.split(pattern, full_text, flags=re.MULTILINE)
cases_list = [c.strip() for c in cases_list if c.strip()]

# convert each case into langchain.schema.Document
case_docs = []
for case_text in cases_list:
    lines = case_text.split("\n", 1)
    title = lines[0].strip() if lines else "UNKNOWN"
    body = lines[1].strip() if len(lines) > 1 else ""
    # use title as metadata
    case_docs.append(Document(page_content=case_text, metadata={"title": title}))

print(case_docs)


data_to_store = []
for doc in case_docs:
    data_to_store.append({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    })

with open("case_docs.json", "w", encoding="utf-8") as f:
    json.dump(data_to_store, f, ensure_ascii=False, indent=2)

print("✅ case_docs have been saved to case_docs.json.")
