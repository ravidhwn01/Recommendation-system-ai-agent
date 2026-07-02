"""Build the persisted Chroma vector store from the raw SHL catalog.

Run from the project root:  python build_vector_db.py
This wipes any existing store and rebuilds it from data/raw/catalog_raw.json.
"""

import shutil
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.data import DataCleaner, DataLoader, DatasetValidator
from app.rag.document_builder import DocumentBuilder
from app.services.vector_service import VectorService


def main() -> None:

    db_path = Path(settings.VECTOR_DB_PATH)

    if db_path.exists():
        logger.info(f"Removing existing vector store at {db_path}")
        shutil.rmtree(db_path)

    raw = DataLoader("data/raw/catalog_raw.json").load()

    validated = DatasetValidator().validate(raw)

    clean = DataCleaner().clean(validated)

    documents = DocumentBuilder().build(clean)

    store = VectorService()
    store.add_documents(documents)

    logger.info(f"Indexed {len(documents)} documents into {db_path}")

    # Sanity check: confirm URLs made it into the metadata.
    sample = store.similarity_search("java developer", k=1)
    if sample:
        logger.info(f"Sample: {sample[0].metadata.get('name')} -> "
                    f"{sample[0].metadata.get('url')}")


if __name__ == "__main__":
    main()
