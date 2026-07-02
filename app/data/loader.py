# from pathlib import Path
# import json

# from core.logger import logger


# class DataLoader:

#     def __init__(self, dataset_path: str):

#         project_root = Path(__file__).resolve().parents[2]

#         self.dataset_path = project_root / dataset_path

#     def load(self):

#         logger.info(f"Loading dataset: {self.dataset_path}")

#         if not self.dataset_path.exists():
#             raise FileNotFoundError(
#                 f"Dataset not found: {self.dataset_path}"
#             )

#         with self.dataset_path.open(
#             "r",
#             encoding="utf-8"
#         ) as file:

#             data = json.load(file)

#         logger.info(f"Loaded {len(data)} records")

#         return data