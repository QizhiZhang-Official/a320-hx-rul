import pandas as pd
import yaml
import numpy as np
import torch

from tqdm import tqdm


class PreProcessor:
    def __init__(self):
        self.selected_qar = self._get_selected_qar()
        self.pack_1_parameters, self.pack_2_parameters = self._get_pack_parameters()

    def _get_selected_qar(self) -> list:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            qar_params = yaml.safe_load(f)
        selected_qar = [item["name"] for item in qar_params["selected_parameters"]]

        return selected_qar

    def _get_pack_parameters(self) -> list | list:
        with open("configs/qar_params.yaml", "r", encoding="utf-8") as f:
            qar_params = yaml.safe_load(f)
        pack_1_parameters = qar_params["pack_1_parameters"]
        pack_2_parameters = qar_params["pack_2_parameters"]

        return pack_1_parameters, pack_2_parameters

    def check_phase(self, data: pd.DataFrame, phase: int) -> bool | pd.DataFrame:
        existing_phases = set(data["Flight Phase from DMU"].dropna().unique())
        required_phases = set(range(2, 10))

        if not required_phases.issubset(existing_phases):
            return False, data
        else:
            data = data[data["Flight Phase from DMU"] == phase].copy()
            return True, data

    def indexing(self, data: pd.DataFrame) -> pd.DataFrame:
        data["time"] = pd.to_datetime(data["time"])
        data["time"] = data["time"].interpolate(method="linear")
        data.set_index("time", inplace=True)

        return data

    def filtering(self, data: pd.DataFrame) -> pd.DataFrame:

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

        return data

    def sync_sample_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.interpolate(method="linear", limit=5)
        data = data.ffill().bfill()

        return data

    def process_one_lifecycle_data(
        self, lifecycle_data: list[dict], scaler, device: str, verbose: bool
    ) -> list[dict]:
        for i in tqdm(range(len(lifecycle_data)), disable=not verbose):
            data = lifecycle_data[i]["data"]
            pack = lifecycle_data[i]["pack"]
            _, data = self.check_phase(data=data, phase=2)
            data = self.indexing(data=data)
            data = self.filtering(data=data)
            data = self.standardize(data=data)
            data = self.sync_sample_rate(data=data)
            if pack == 1:
                data = data[self.pack_1_parameters].copy()
            if pack == 2:
                data = data[self.pack_2_parameters].copy()
            seq_len = len(data)
            data = scaler.transform(data=data.values)
            x = torch.tensor(data, dtype=torch.float32, device=device)
            x.unsqueeze(0)
            padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            lifecycle_data[i]["x"] = x
            lifecycle_data[i]["padding_mask"] = padding_mask
            lifecycle_data[i].pop("data")

        return lifecycle_data

    def exec(
        self, craft_data: list[dict[str, str | pd.DataFrame]], phase: int
    ) -> list[dict[str, str | pd.DataFrame]]:
        processed_craft_data = []

        for item in tqdm(craft_data, desc="--"):
            file_name = item["file_name"]
            data = item["data"]

            is_phase_full, data = self.check_phase(data, phase)
            if is_phase_full == False:
                # print(f"{file_name} 8个阶段不完整")
                continue

            data = self.indexing(data=data)

            data = self.filtering(data=data)

            data = self.standardize(data=data)

            data = self.sync_sample_rate(data=data)

            processed_craft_data.append({"file_name": file_name, "data": data})

        return processed_craft_data
