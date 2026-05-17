# src/data/dataset.py

import os
import yaml
import torch
import numpy as np
import pandas as pd
from src.data.preprocess import PreProcessor
from torch.utils.data import Dataset


class HXRULDataset(Dataset):
    def __init__(self, raw_data_dir: str):
        super().__init__()
        self.annotations = self._load_annotations()
        self.flattened_annotations = self._flatten_annotations()
        self.feature_cols = self._load_feature_params()
        self.raw_data_dir = raw_data_dir

    def __len__(self) -> int:

        return len(self.flattened_annotations)

    def __getitem__(self, index) -> dict:
        flight = self.flattened_annotations[index]
        file_path = os.path.join(
            self.raw_data_dir,
            str(flight["lifecycle_start_date"].year),
            flight["craft_no"],
            flight["file_name"],
        )
        data = pd.read_csv(file_path, dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str})
        data = self._feature_fn(data, flight["pack"])

        x = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(flight["rul_h"], dtype=torch.float32)
        mask = torch.ones(x.size(0), dtype=torch.float32)

        return {
            "features": x,
            "rul": y,
            "mask": mask,
            "meta": {
                "craft_no": flight["craft_no"],
                "lifecycle_id": flight["lifecycle_id"],
                "sub_id": flight["sub_id"],
                "pack": flight["pack"],
                "duration_s": flight["duration_s"],
            },
        }

    def _load_annotations(self) -> list:
        with open("configs/annotations.yaml", "r") as f:
            annotations = yaml.safe_load(f)

        return annotations

    def _flatten_annotations(self) -> list:
        flattened_annotations = []
        for item in self.annotations:
            flattened_annotations += item["annotation"]

        return flattened_annotations

    def _load_feature_params(self) -> dict:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            qar_params = yaml.safe_load(f)
        features_pack_1 = qar_params["pack_1_parameters"]
        features_pack_2 = qar_params["pack_2_parameters"]
        features = {
            1: features_pack_1,
            2: features_pack_2,
        }

        return features

    def _feature_fn(self, data: pd.DataFrame, pack: int) -> pd.DataFrame:
        preprocessor = PreProcessor()
        is_phase_full, data = preprocessor.check_phase(data, phase=2)
        data = preprocessor.indexing(data)
        data = preprocessor.filtering(data)
        data = preprocessor.standardize(data)
        data = preprocessor.sync_sample_rate(data)

        if pack == 1:
            col_mask = self.feature_cols[1]

            return data[col_mask]

        if pack == 2:
            col_mask = self.feature_cols[2]

            return data[col_mask]
