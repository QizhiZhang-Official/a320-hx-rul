# src/utils/scaler.py
import os
import torch
import numpy as np
import yaml
import pandas as pd
from sklearn.preprocessing import RobustScaler
from src.data.preprocess import PreProcessor


class DataScaler:
    def __init__(self, raw_data_dir: str, use_sample: int):
        self.raw_data_dir = raw_data_dir
        self.use_sample = use_sample
        self.scaler = None
        self.is_fitted = False
        
        all_data_np = self.get_data_for_fit()
        self.fit(all_data_np)
        self.save('scaler.pkl')

    def load_annotations(self) -> list:
        with open("configs/annotations.yaml", "r") as f:
            annotations = yaml.safe_load(f)
        
        return annotations
    
    def load_pack_parameters(self) -> list | list:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            qar_params = yaml.safe_load(f)
            pack_1_parameters = qar_params["pack_1_parameters"]
            pack_2_parameters = qar_params["pack_2_parameters"]
        
        return pack_1_parameters, pack_2_parameters
        
    def get_data_for_fit(self) -> np.ndarray:
        data_to_fit_scaler = []
        preprocessor = PreProcessor()
        annotations = self.load_annotations()
        pack_1_parameters, pack_2_parameters = self.load_pack_parameters()
        for i in range(self.use_sample):
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
                _, data = preprocessor.check_phase(data)
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

    def fit(self, all_data: np.ndarray) -> None:
        self.scaler = RobustScaler()
        self.scaler.fit(all_data)
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.is_fitted == True, "Scaler has not been fitted."
        return self.scaler.transform(data)

    def save(self, name: str) -> None:
        save_path = os.path.join("checkpoints", name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.scaler, save_path)

    def load(self, name: str) -> None:
        load_path = os.path.join("checkpoints", name)
        assert os.path.exists(load_path), f"{name} not found."
        self.scaler = torch.load(load_path, map_location="cpu")
        self.is_fitted = True
