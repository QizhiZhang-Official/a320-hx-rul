import pandas as pd
import os
import yaml

from tqdm import tqdm


def load_an_aircraft(craft_no: str) -> list[dict[str, str | pd.DataFrame]]:
    DIR_2023 = r"D:/2022年  国航CBM关键技术研究项目/20220616 甘工提供的数据和资料/第6批  202405 A320 2023年数据/320CFM/"
    DIR_2024 = r"D:/2022年  国航CBM关键技术研究项目/20220616 甘工提供的数据和资料/第8批数据  A320 2024年总表/2024-320/A320/"

    craft_data = []

    if craft_no in os.listdir(DIR_2023):
        for file_name in tqdm(
            os.listdir(os.path.join(DIR_2023, craft_no)), desc="--2023"
        ):
            file_path = os.path.join(DIR_2023, craft_no, file_name)
            if os.path.getsize(file_path) > 0:
                flight_data = {
                    "file_name": file_name,
                    "data": pd.read_csv(
                        file_path,
                        dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
                    ),
                }
                craft_data.append(flight_data)

    if craft_no in os.listdir(DIR_2024):
        for file_name in tqdm(
            os.listdir(os.path.join(DIR_2024, craft_no)), desc="--2024"
        ):
            file_path = os.path.join(DIR_2024, craft_no, file_name)
            if os.path.getsize(file_path) > 0:
                flight_data = {
                    "file_name": file_name,
                    "data": pd.read_csv(
                        file_path,
                        dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
                    ),
                }
                craft_data.append(flight_data)

    return craft_data


def load_zipped_data(
    load_dir: str, verbose: bool = True
) -> list[dict[str, str | pd.DataFrame]]:
    zipped_data = []
    for craft_no in tqdm(os.listdir(load_dir), desc="载入数据", disable=not verbose):
        data = pd.read_csv(os.path.join(load_dir, craft_no))
        data.set_index("time", inplace=True)
        flight_data = {"craft_no": craft_no.replace(".csv", ""), "data": data}
        zipped_data.append(flight_data)

    return zipped_data


class DataIO:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.annotations = self._load_annotations()
        self.all_lifecycle_id = self._load_all_lifecycle_id()
        pass

    def _load_annotations(self) -> list:
        with open("configs/annotations.yaml", "r") as f:
            annitations = yaml.safe_load(f)

        return annitations

    def _load_all_lifecycle_id(self) -> list:
        with open("configs/all_lifecycle_id.yaml", "r") as f:
            all_life_cycle_id = yaml.safe_load(f)

        return all_life_cycle_id

    def get_one_lifecycle(self, lifecycle_id: int, verbose: bool) -> list:
        lifecycle = self.annotations[self.all_lifecycle_id.index(lifecycle_id)]
        lifecycle_craft_no = lifecycle["craft_no"]
        lifecycle_year = str(lifecycle["start_date"].year)
        lifecycle_data = lifecycle["annotation"]
        for i in tqdm(range(len(lifecycle_data)), disable=not verbose):
            data_path = os.path.join(
                self.raw_data_dir,
                lifecycle_year,
                lifecycle_craft_no,
                lifecycle_data[i]["file_name"],
            )
            data = pd.read_csv(
                data_path,
                dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
            )
            lifecycle_data[i]['data'] = data
        
        return lifecycle_data
    
    def save_one_encoded_lifecycle(self, lifecycle_data: list[dict], save_path: str, verbose: bool) -> None:
        rows = []
        for flight in tqdm(lifecycle_data, disable=not verbose):
            row = {
                'sub_id': flight['sub_id'],
                'file_name': flight['file_name'],
                'flight_no': flight['flight_no'],
                'duration_s': flight['duration_s'],
                'rul_s': flight['rul_s'],
                'rul_h': flight['rul_h']
            }
            y = flight['y'].detach().cpu().numpy()
            for i, value in enumerate(y):
                row[f'feature_{i}'] = value
            rows.append(row)
        
        # cols = ['sub_id', 'file_name', 'flight_no', 'duration_s', 'rul_s', 'rul_h'] + [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(rows)
        data.to_csv(save_path, index=False)
