import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.cleaner import DataCleaner
from app.data.loader import DataLoader
from app.data.validator import DatasetValidator

from app.rag.document_builder import DocumentBuilder


loader = DataLoader("data/raw/catalog_raw.json")

raw = loader.load()

validator = DatasetValidator()

validated = validator.validate(raw)

cleaner = DataCleaner()

clean = cleaner.clean(validated)

builder = DocumentBuilder()

documents = builder.build(clean)

print(len(documents))