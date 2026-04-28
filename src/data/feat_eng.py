import yaml
import pandas as pd
from tqdm import tqdm


class FeatProcessor:
    def __init__(self):
        self.feat_map = self._load_feat_map()

    def _load_feat_map(self) -> dict:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        feat_map = {}
        selected_qar = config["selected_parameters"]
        for item in selected_qar:
            feat_map[item["name"]] = item["map"]

        return feat_map

    def symbolize_feat(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.rename(columns=self.feat_map)

        return data

    def calculate_PrHX_eff(self, data: pd.DataFrame) -> pd.DataFrame:
        data["PrHX_1_eff"] = (data["TPO_1"] - data["COT_1"]) / (
            data["TPO_1"] - data["SAT"] + 1e-8
        )
        data["PrHX_2_eff"] = (data["TPO_2"] - data["COT_2"]) / (
            data["TPO_2"] - data["SAT"] + 1e-8
        )

        return data

    def exec(
        self, craft_data: list[dict[str, str | pd.DataFrame]]
    ) -> list[dict[str, str | pd.DataFrame]]:
        processed_craft_data = []

        for item in tqdm(craft_data, desc='--'):
            file_name = item["file_name"]
            data = item["data"]

            data = self.symbolize_feat(data)
            data = self.calculate_PrHX_eff(data)

            processed_craft_data.append({"file_name": file_name, "data": data})

        return processed_craft_data
