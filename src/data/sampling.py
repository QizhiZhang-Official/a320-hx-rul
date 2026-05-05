import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_io import load_zipped_data


class Sampler():
    def __init__(self, zipped_data_dir: str):
        self.zipped_data_dir = zipped_data_dir
        self.extracted_data = None
        self.extracted_pack = None
    
    def extract_from_zipped(self, craft_no: str, start_time: str, end_time: str, pack: int, modify: int):
        zipped_data = load_zipped_data(self.zipped_data_dir)
        for item in zipped_data:
            if craft_no != item["craft_no"]:
                continue
            data = item["data"]
            
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            period_data = data.loc[start_time:end_time]
            if modify > 0:
                modified_data = period_data.iloc[:-modify]
            else:
                modified_data = period_data
        
        self.extracted_data = modified_data
        self.extracted_pack = pack
    
    def plot_extracted(self):
        plt.figure()
        plt.plot(self.extracted_data.index, self.extracted_data[f"COT_{str(self.extracted_pack)}_mean"])
        plt.xlabel("Time")
        plt.ylabel(f"COT_{str(self.extracted_pack)}_mean")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def get_all_file_name_list(self):
        extracted_data = self.extracted_data
        if extracted_data == None:
            return None
        
        extracted_data.sort_index(inplace=True)
        
        return extracted_data['file_name'].tolist()
    
    


def sample(
    zipped_data_dir: str,
    craft_no: str,
    start_time: str,
    end_time: str,
    pack: int,
    modify: int,
):
    zipped_data = load_zipped_data(zipped_data_dir)
    for item in zipped_data:
        if craft_no != item["craft_no"]:
            continue
        data = item["data"]
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        period_data = data.loc[start_time:end_time]
        if modify > 0:
            modified_data = period_data.iloc[:-modify]
        else:
            modified_data = period_data
        
        plt.figure()
        plt.plot(modified_data.index, modified_data[f"COT_{str(pack)}_mean"])
        plt.xlabel("Time")
        plt.ylabel(f"COT_{str(pack)}_mean")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()