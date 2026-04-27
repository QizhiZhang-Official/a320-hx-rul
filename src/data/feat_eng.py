import yaml
import pandas as pd


class FeatProcessor:
    def __init__(self):
        self.feat_map = self._load_feat_map()

    def _load_feat_map(self) -> dict:
        with open("configs/qar_params.yaml", "r") as f:
            config = yaml.safe_load(f)
        feat_map = {}
        selected_qar = config["selected_parameters"]
        for item in selected_qar:
            feat_map = {item["name"]: item["map"]}

        return feat_map

    def symbolize_feat(self, data: pd.DataFrame) -> pd.DataFrame:
        data.rename(columns=self.feat_map)

        return data

    def calculate_PrHX_eff(self, data: pd.DataFrame) -> pd.DataFrame:
        data["PrHX_1_eff"] = (data["TPO_1"] - data["COT_1"]) / (
            data["TPO_1"] - data["SAT"] + 1e-8
        )
        data["PrHX_2_eff"] = (data["TPO_2"] - data["COT_2"]) / (
            data["TPO_2"] - data["SAT"] + 1e-8
        )

        return data

    def exec(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.symbolize_feat(data)
        data = self.calculate_PrHX_eff(data)

        return data
