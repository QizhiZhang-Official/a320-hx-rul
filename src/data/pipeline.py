import yaml
import pandas as pd
import os
from src.data.data_io import *
from src.data.preprocess import PreProcessor
from src.data.feat_eng import FeatProcessor
from src.data.feat_zip import FeatZipper

def process_one_craft(craft_no: str, phase: int, save_dir: str):
    with open("configs/all_craft_no.yaml", "r") as f:
        all_craft_no = yaml.safe_load(f)

    preprocessor = PreProcessor()
    feat_processor = FeatProcessor()
    feat_zipper = FeatZipper("only_mean")
    
    print(f"\n处理{craft_no}")
    print("-载入数据...")
    craft_data = load_an_aircraft(craft_no)
    print("-预处理...")
    craft_data = preprocessor.exec(craft_data, phase)
    print("-特征工程...")
    craft_data = feat_processor.exec(craft_data)
    print("-统计...")
    craft_data_df = feat_zipper.exec(craft_data)
    print("-保存...")
    craft_data_df['time'] = pd.to_datetime(craft_data_df['time'])
    craft_data_df.set_index('time', inplace=True)
    craft_data_df.sort_index(inplace=True)
    craft_data_df.reset_index(inplace=True)
    os.makedirs(save_dir, exist_ok=True)
    craft_data_df.to_csv(os.path.join(save_dir, craft_no + ".csv"), index=False)


def step_1(phase: int, save_dir: str):
    with open("configs/all_craft_no.yaml", "r") as f:
        all_craft_no = yaml.safe_load(f)

    preprocessor = PreProcessor()
    feat_processor = FeatProcessor()
    feat_zipper = FeatZipper("only_mean")

    for craft_no in all_craft_no:
        print(f"\n处理{craft_no}")
        print("-载入数据...")
        craft_data = load_an_aircraft(craft_no)
        print("-预处理...")
        craft_data = preprocessor.exec(craft_data, phase)
        print("-特征工程...")
        craft_data = feat_processor.exec(craft_data)
        print("-统计...")
        craft_data_df = feat_zipper.exec(craft_data)
        print("-保存...")
        craft_data_df['time'] = pd.to_datetime(craft_data_df['time'])
        craft_data_df.set_index('time', inplace=True)
        craft_data_df.sort_index(inplace=True)
        craft_data_df.reset_index(inplace=True)
        os.makedirs(save_dir, exist_ok=True)
        craft_data_df.to_csv(os.path.join(save_dir, craft_no + ".csv"), index=False)


def generate_annotations(zipped_data_dir: str):
    with open("configs/extract_config.yaml", "r") as f:
        extract_config = yaml.safe_load(f)
    
    zipped_data = load_zipped_data(zipped_data_dir)
    for item in tqdm(extract_config):
        craft_no = item["craft_no"]
        start_date = item["start_date"]
        end_date = item["end_date"]
        pack = item["pack"]
        modify = item["modify"]

        for craft_data in zipped_data:
            if craft_data['craft_no'] != craft_no:
                continue
            data = craft_data["data"]
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            period_data = data.loc[start_date:end_date]
            if modify > 0:
                modified_data = period_data.iloc[:-modify]
            else:
                modified_data = period_data
            modified_data = modified_data.sort_index()
        