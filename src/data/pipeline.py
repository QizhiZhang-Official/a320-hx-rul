import yaml
import pandas as pd
import os
from src.data.data_io import load_an_aircraft
from src.data.preprocess import PreProcessor
from src.data.feat_eng import FeatProcessor
from src.data.feat_zip import FeatZipper

def process_one_craft(craft_no: str, save_dir: str):
    with open("configs/all_craft_no.yaml", "r") as f:
        all_craft_no = yaml.safe_load(f)

    preprocessor = PreProcessor()
    feat_processor = FeatProcessor()
    feat_zipper = FeatZipper("only_mean")
    
    print(f"\n处理{craft_no}")
    print("-载入数据...")
    craft_data = load_an_aircraft(craft_no)
    print("-预处理...")
    craft_data = preprocessor.exec(craft_data)
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


def step_1(save_dir: str):
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
        craft_data = preprocessor.exec(craft_data)
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
