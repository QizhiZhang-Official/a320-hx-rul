import yaml
import pandas as pd
import os
import torch
from src.data.data_io import *
from src.data.preprocess import PreProcessor
from src.data.feat_eng import FeatProcessor
from src.data.feat_zip import FeatZipper
from src.data.anno_gen import AnnoGenerator
from src.models.encoder import Encoder
from src.data.dataloader import collate_fn
from src.utils.scaler import DataScaler


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
    craft_data_df["time"] = pd.to_datetime(craft_data_df["time"])
    craft_data_df.set_index("time", inplace=True)
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
        craft_data_df["time"] = pd.to_datetime(craft_data_df["time"])
        craft_data_df.set_index("time", inplace=True)
        craft_data_df.sort_index(inplace=True)
        craft_data_df.reset_index(inplace=True)
        os.makedirs(save_dir, exist_ok=True)
        craft_data_df.to_csv(os.path.join(save_dir, craft_no + ".csv"), index=False)


def generate_annotations(
    raw_data_dir: str, zipped_data_dir: str, PF_threshold: float, max_rul_s: int
) -> None:
    print("\n生成 annotations.yaml")
    anno_generator = AnnoGenerator(
        raw_data_dir, zipped_data_dir, PF_threshold, max_rul_s
    )
    anno_generator.exec()


def encode_rul_dataset(
    raw_data_dir: str, save_dir: str, checkpoint: str, scaler_name: str, device: str
) -> None:
    with open("configs/all_lifecycle_id.yaml", "r") as f:
        all_lifecycle_id = yaml.safe_load(f)
    data_io = DataIO(raw_data_dir=raw_data_dir)
    preprocessor = PreProcessor()

    print("Initializing Scaler...")
    scaler = DataScaler.load(name=scaler_name)

    print("Initializing Encoder...")
    encoder = Encoder.load_checkpoint(
        checkpoint_path=os.path.join("checkpoints", checkpoint), device=device
    )

    for lifecycle_id in all_lifecycle_id:
        print(f"\nLifecycle_id: {lifecycle_id}")

        print("Loading Data...")
        lifecycle_data = data_io.get_one_lifecycle(
            lifecycle_id=lifecycle_id, verbose=True
        )

        print("Preprocessing...")
        lifecycle_data = preprocessor.process_one_lifecycle_data(
            lifecycle_data=lifecycle_data, scaler=scaler, device=device, verbose=True
        )

        print("Encoding...")
        lifecycle_data = encoder.encode_one_lifecycle(
            lifecycle_data=lifecycle_data, verbose=True
        )

        print("Saving...")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, lifecycle_id + ".csv")
        data_io.save_one_encoded_lifecycle(
            lifecycle_data=lifecycle_data, save_path=save_path, verbose=True
        )
