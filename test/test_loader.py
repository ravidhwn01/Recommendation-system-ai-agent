from app.data import DataCleaner
from app.data import DataLoader
from app.data import DatasetValidator

loader = DataLoader(
    "data/raw/catalog_raw.json"
)

raw = loader.load()

validator = DatasetValidator()

validated = validator.validate(raw)

cleaner = DataCleaner()

clean = cleaner.clean(validated)

print(clean)