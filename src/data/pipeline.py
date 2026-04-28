import yaml
import os
from src.data.io import load_an_aircraft
from src.data.preprocess import PreProcessor
from src.data.feat_eng import FeatProcessor
from src.data.feat_zip import FeatZipper


def step_1(save_dir: str):
    with open("configs/all_craft_no.yaml", "r") as f:
        all_craft_no = yaml.safe_load(f)
    with open("configs/zip_method.yaml", "r") as f:
        zip_method = yaml.safe_load(f)
    method_config = zip_method["only_mean"]

    preprocessor = PreProcessor()
    feat_processor = FeatProcessor()
    feat_zipper = FeatZipper(method_config)

    for craft_no in all_craft_no:
        print(f'\n处理{craft_no}')
        craft_data = load_an_aircraft(craft_no)
        print(' 载入数据...')
        craft_data = preprocessor.exec(craft_data)
        print(' 预处理...')
        craft_data = feat_processor.exec(craft_data)
        print(' 特征工程...')
        craft_data_df = feat_zipper.exec(craft_data)
        print(' 统计...')

        os.makedirs(save_dir, exist_ok=True)
        craft_data_df.to_csv(os.path.join(save_dir, craft_no + ".csv"), index=False)
