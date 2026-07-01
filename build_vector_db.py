from app.data import DataCleaner
from app.data import DataLoader
from app.data import DatasetValidator

from app.rag.document_builder import DocumentBuilder
from app.rag.vector_store import VectorStore


loader = DataLoader("data/raw/catalog_raw.json")

raw = loader.load()

validator = DatasetValidator()

validated = validator.validate(raw)

cleaner = DataCleaner()

clean = cleaner.clean(validated)

builder = DocumentBuilder()

documents = builder.build(clean)

db = VectorStore()

db.add_documents(documents)

print(f"Indexed {len(documents)} documents.")