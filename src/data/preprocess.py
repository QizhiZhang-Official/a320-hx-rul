import pandas as pd
import yaml
import numpy as np


class PreProcessor:
    def __init__(self):
        self.selected_qar = self._get_selected_qar()

    def _get_selected_qar(self) -> list:
        with open("configs/qar_params.yaml", "r") as f:
            config = yaml.safe_load(f)
        selected_qar = [item["name"] for item in config["selected_parameters"]]

        return selected_qar

    def check_phase(self, data: pd.DataFrame) -> bool:
        existing_phases = set(data["Flight Phase from DMU"].dropna().unique())
        required_phases = set(range(2, 10))

        return required_phases.issubset(existing_phases)

    def check_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        if data["time"].isna().any():
            return False, data
        else:
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
            return True, data

    def filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data["Flight Phase from DMU"] == 6].copy()
        data = data[self.selected_qar].copy()

        return data

    def standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        data["PACK WATER EXTR.TEMPRATURE SYS.1"] = data[
            "PACK WATER EXTR.TEMPRATURE SYS.1"
        ].replace(0, np.nan)
        data["PACK WATER EXTR.TEMPRATURE SYS.2"] = data[
            "PACK WATER EXTR.TEMPRATURE SYS.2"
        ].replace(0, np.nan)

        data["Eng 1 PRV not fully close"] = data["Eng 1 PRV not fully close"].map(
            {"FULLY CLOSED": 0, "NOT FULLY CLOSED": 1}
        )
        data["Eng 2 PRV not fully close"] = data["Eng 2 PRV not fully close"].map(
            {"FULLY CLOSED": 0, "NOT FULLY CLOSED": 1}
        )

    def sync_sample_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = pd.to_numeric(data, errors="coerce")
        data = data.interpolate(method="linear", limit=5)
        data = data.ffill().bfill()

        return data

    def exec(
        self, craft_data: list[dict[str, str | pd.DataFrame]]
    ) -> list[dict[str, str | pd.DataFrame]]:
        processed_craft_data = []

        for item in craft_data:
            file_name = item["file_name"]
            data = item["data"]

            if self.check_phase(data) == False:
                print(f"{file_name}8个阶段不完整")
                continue

            is_datetime_full, data = self.check_datetime(data)
            if is_datetime_full == False:
                raise Exception(f"{file_name}日期有缺失")

            data = self.filtering(data)

            data = self.standardize(data)

            data = self.sync_sample_rate(data)

            processed_craft_data.append({"file_name": file_name, "data": data})

        return processed_craft_data
