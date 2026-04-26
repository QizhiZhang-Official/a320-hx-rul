import pandas as pd
import yaml

def check_phase(data: pd.DataFrame) -> bool:
    is_data_available = True

    for phase in range(2, 10):
        if phase not in data["Flight Phase from DMU"].values:
            is_data_available = False
            break

    return is_data_available


def check_datetime(data: pd.DataFrame) -> pd.DataFrame:
    if data["time"].isna().any():
        return False, data
    else:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        return True, data


def filtering():
    data = data[data["Flight Phase from DMU"] == 6].copy()
    
    with open('configs/qar_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    selected_mask = [item['name'] for item in config['selected_parameters']]
    data = data[selected_mask].copy()
    
    return data


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    data['PACK WATER EXTR.TEMPRATURE SYS.1'] = data['PACK WATER EXTR.TEMPRATURE SYS.1'].replace(0, np.nan)
    data['PACK WATER EXTR.TEMPRATURE SYS.2'] = data['PACK WATER EXTR.TEMPRATURE SYS.2'].replace(0, np.nan)
    
    data['Eng 1 PRV not fully close'] = data['Eng 1 PRV not fully close'].map{'FULLY CLOSED': 0, 'NOT FULLY CLOSED': 1}
    data['Eng 2 PRV not fully close'] = data['Eng 2 PRV not fully close'].map{'FULLY CLOSED': 0, 'NOT FULLY CLOSED': 1}
    


def sync_sample_rate(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.to_numeric(data, errors='coerce')
    data = data.interpolate(method='linear', limit=5)
    data = data.ffill().bfill()
    
    return data