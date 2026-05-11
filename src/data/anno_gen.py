import pandas as pd
import yaml
import os
class AnnoGenerator():
    def __init__(self, raw_data_dir: str, zipped_data_dir: str, PF_threshold: float):
        self.raw_data_dir = raw_data_dir
        self.zipped_data_dir = zipped_data_dir
        self.PF_threshold = PF_threshold
        self.zipped_data = self._load_zipped_data()
        self.extract_config = self._load_extract_config()
        
    
    def _load_extract_config(self):
        with open('configs/extract_config.yaml', 'r') as f:
            extract_config = yaml.safe_load(f)
        
        return extract_config
    
    def _load_zipped_data(self):
        from src.data.data_io import load_zipped_data
        zipped_data = load_zipped_data(self.zipped_data_dir, verbose=False)
        
        return zipped_data
    
    def _get_flight_no_from_file_name(self, file_name: str) -> str:
        for str in file_name.split('_'):
            if 'CA' in str:
                flight_no = str.replace('.csv', '')
                return flight_no
        
        return '-------'
    
    def _get_duration(self, data: pd.DataFrame, pack: int, PF_threshold: float) -> int:
        if pack == 1:
            col_name = 'PACK FLOW SYS.1'
        if pack == 2:
            col_name = 'PACK FLOW SYS.2'
        data_clip = data[data[col_name] >= PF_threshold]
        duration = len(data_clip)
        
        return duration
    
    def _get_file_name_list(self, extr_conf_dic: dict) -> list:
        craft_no = extr_conf_dic['craft_no']
        start_date = extr_conf_dic['start_date']
        end_date = extr_conf_dic['end_date']
        pack = extr_conf_dic['pack']
        modify = extr_conf_dic['modify']
        
        for craft_data in self.zipped_data:
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
            file_name_list = modified_data['file_name'].tolist()
            
            return file_name_list
            
    
    def _get_annotation(self, extr_conf_dic: dict) -> list[dict]:
        annotation = []
        file_name_list = self._get_file_name_list(extr_conf_dic)
        for file_name in file_name_list:
            craft_no = extr_conf_dic['craft_no']
            start_date = extr_conf_dic['start_date']
            pack = extr_conf_dic['pack']
            year = start_date.year
            data_path = os.path.join(self.raw_data_dir, str(year), craft_no, file_name)
            data = pd.read_csv(data_path)
            
            flight_anno = {}
            flight_anno['file_name'] = file_name
            flight_anno['flight_no'] = self._get_flight_no_from_file_name(file_name)
            flight_anno['duration'] = self._get_duration(data, pack, self.PF_threshold)
            flight_anno['rul'] = None
            
            annotation.append(flight_anno)
        
        return annotation
    
    def _save_as_yaml(self, annotations: list) -> None:
        with open('configs/annotations.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(annotations, f, default_flow_style=False)
    
    def exec(self):
        annotations = []
        
        from tqdm import tqdm
        for extr_conf_dic in tqdm(self.extract_config, desc='-'):
            raw_dic = extr_conf_dic
            annotation = self._get_annotation(extr_conf_dic)
            raw_dic['annotation'] = annotation
            
            annotations.append(raw_dic)
        
        self._save_as_yaml(annotations)
        