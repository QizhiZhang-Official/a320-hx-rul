import pandas as pd
import os
from tqdm import tqdm


def load_an_aircraft(craft_no: str) -> list[dict[str, str | pd.DataFrame]]:
    DIR_2023 = r"D:/2022年  国航CBM关键技术研究项目/20220616 甘工提供的数据和资料/第6批  202405 A320 2023年数据/320CFM/"
    DIR_2024 = r"D:/2022年  国航CBM关键技术研究项目/20220616 甘工提供的数据和资料/第8批数据  A320 2024年总表/2024-320/A320/"

    craft_data = []

    if craft_no in os.listdir(DIR_2023):
        for file_name in tqdm(os.listdir(os.path.join(DIR_2023, craft_no)), desc="--2023"):
            flight_data = {
                "file_name": file_name,
                "data": pd.read_csv(
                    os.path.join(DIR_2023, craft_no, file_name),
                    dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
                ),
            }
            craft_data.append(flight_data)

    if craft_no in os.listdir(DIR_2024):
        for file_name in tqdm(os.listdir(os.path.join(DIR_2024, craft_no)), desc="--2024"):
            flight_data = {
                "file_name": file_name,
                "data": pd.read_csv(
                    os.path.join(DIR_2024, craft_no, file_name),
                    dtype={"CITY_PAIR_FR": str, "CITY_PAIR_TO": str},
                ),
            }
            craft_data.append(flight_data)

    return craft_data
