import yaml
import pandas as pd


class FeatZipper:
    def __init__(self, method_config: str):
        self.zip_method = self._load_zip_method(method_config)

    def _load_zip_method(self, method_config: str) -> list:
        with open("configs/zip_method.yaml", "r") as f:
            config = yaml.safe_load(f)
        zip_method = config[method_config]

        return zip_method
    
    def zip_feat(self, data: pd.DataFrame)

    def zip_feat_to_dict(self, data: pd.DataFrame) -> dict:
        flight_dict = {"time": data["time"][0]}
        for method in self.zip_method:
            if method == 'mean':
                flight_dict = {}
