import yaml
import pandas as pd
from tqdm import tqdm


class FeatZipper:
    def __init__(self, method_config: str):
        self.zip_method = self._load_zip_method(method_config)

    def _load_zip_method(self, method_config: str) -> list[str]:
        with open("configs/zip_method.yaml", "r") as f:
            config = yaml.safe_load(f)
        zip_method = config[method_config]

        return zip_method

    def zip_feat_to_dict(self, data: pd.DataFrame) -> dict:
        flight_dict = {"time": data["time"][0]}
        for feat in data.columns.to_list():
            for method in self.zip_method:
                if method == "mean":
                    flight_dict[feat + "_mean"] = data[feat].mean()
                elif method == "std":
                    flight_dict[feat + "_std"] = data[feat].std()
                else:
                    raise Exception("未知的zip方法")

        return flight_dict

    def exec(self, craft_data: list[dict[str, str | pd.DataFrame]]) -> pd.DataFrame:
        processed_craft_data = []

        for item in tqdm(craft_data, desc='--'):
            file_name = item["file_name"]
            data = item["data"]

            zipped_data = self.zip_feat_to_dict(data)

            processed_craft_data.append(zipped_data)

        zipped_craft_data_df = pd.DataFrame(processed_craft_data)

        return zipped_craft_data_df
