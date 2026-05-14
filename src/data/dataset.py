import yaml
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class HXRULDataset(Dataset):
    def __init__(self, raw_data_dir: str):
        super().__init__()
        self.annotations = self._load_annotations()
        self.feature_cols = self._load_feature_params()
        self.raw_data_dir = raw_data_dir

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        features = self

    def _load_annotations(self) -> list:
        with open("configs/annotations.yaml", "r") as f:
            annotations = yaml.safe_load(f)

        return annotations

    def _load_feature_params(self) -> dict:
        with open("configs/qar_params.yaml", "r") as f:
            qar_params = yaml.safe_load(f)
        features_pack_1 = qar_params["pcck_1_parameters"]
        features_pack_2 = qar_params["pacl_2_parameters"]
        features = {
            1: features_pack_1,
            2: features_pack_2,
        }

        return features

    def _feature_fn(self, data: pd.DataFrame, pack: int) -> pd.DataFrame:
        if pack == 1:
            col_mask = self.feature_cols[1]
            
            return data[col_mask]
        
        if pack == 2:
            col_mask = self.feature_cols[2]
            
            return data[col_mask]
