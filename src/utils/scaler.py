# src/utils/scaler.py
import os
import torch
import numpy as np
import yaml
import pandas as pd
import joblib

from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from src.data.preprocess import PreProcessor


class DataScaler:
    def __init__(self, raw_data_dir: str, use_sample: int):
        self.raw_data_dir = raw_data_dir
        self.use_sample = use_sample
        self.scaler = None
        self.is_fitted = False

    def _load_annotations(self) -> list:
        with open("configs/annotations.yaml", "r") as f:
            annotations = yaml.safe_load(f)

        return annotations

    def _load_pack_parameters(self) -> list | list:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            qar_params = yaml.safe_load(f)
            pack_1_parameters = qar_params["pack_1_parameters"]
            pack_2_parameters = qar_params["pack_2_parameters"]

        return pack_1_parameters, pack_2_parameters

    def _get_data_for_fit(self) -> np.ndarray:
        data_to_fit_scaler = []
        preprocessor = PreProcessor()
        annotations = self._load_annotations()
        pack_1_parameters, pack_2_parameters = self._load_pack_parameters()
        for i in tqdm(range(self.use_sample)):
            annotation = annotations[i]["annotation"]
            for flight in annotation:
                year = str(flight["lifecycle_start_date"].year)
                craft_no = flight["craft_no"]
                file_name = flight["file_name"]
                pack = flight["pack"]
                data = pd.read_csv(
                    os.path.join(self.raw_data_dir, year, craft_no, file_name),
                    dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
                )
                _, data = preprocessor.check_phase(data, phase=2)
                data = preprocessor.indexing(data)
                data = preprocessor.filtering(data)
                data = preprocessor.standardize(data)
                data = preprocessor.sync_sample_rate(data)
                if pack == 1:
                    data = data[pack_1_parameters].copy()
                if pack == 2:
                    data = data[pack_2_parameters].copy()
                data_to_fit_scaler.append(data)

        all_data_np = np.vstack(data_to_fit_scaler)

        return all_data_np

    def fit(self) -> None:
        all_data = self._get_data_for_fit()
        self.scaler = RobustScaler()
        self.scaler.fit(all_data)
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.is_fitted == True, "Scaler has not been fitted."
        return self.scaler.transform(data)

    def save(self, name: str) -> None:
        os.makedirs("checkpoints/", exist_ok=True)
        joblib.dump(self.scaler, f"checkpoints/{name}")

    @classmethod
    def load(cls, name: str) -> None:
        assert os.path.exists(f"checkpoints/{name}"), f"{name} not found."
        scaler = cls(raw_data_dir=None, use_sample=None)
        scaler.scaler = joblib.load(f"checkpoints/{name}")
        scaler.is_fitted = True
        
        return scaler
