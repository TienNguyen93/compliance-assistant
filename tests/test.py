# test PyPDFLoader vs pdfplumber for loading PDF files

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pdfplumber

file_path = "data/raw/fda_guidance/GFI106.pdf"
with pdfplumber.open(file_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

print(text)


# loader = PyPDFLoader(file_path)
# pages = loader.load()

# cleaned_pages = []

# for page in pages:
#     if len(page.page_content.split(" ")) > 20:
#         cleaned_pages.append(page)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# schema = {
#     "properties": {
#         "title": {"type": "string"},
#         "keywords": {"type": "array", "items": {"type": "string"}},
#         "hasCode": {"type": "boolean"},
#     },
#     "required": ["title", "keywords", "hasCode"],
# }

# import pdfplumber
# with pdfplumber.open(file_path) as pdf:
#     text = "\n".join(page.extract_text() for page in pdf.pages)