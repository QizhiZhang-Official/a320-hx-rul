import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def plot_PrHX_eff(zipped_data_dir: str, save_dir: str) -> None:
    from src.data.data_io import load_zipped_data
    zipped_data = load_zipped_data(zipped_data_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 定义目标时间段：4月1日 到 8月31日
    START_DATE = "01-01"
    END_DATE = "10-31"

    for item in tqdm(zipped_data, desc='绘图'):
        craft_no = item['craft_no']
        data = item['data']

        # 确保索引为 datetime 类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # 按年份拆分并筛选 4.1-8.31 时间段
        for year in ['2023', '2024']:
            # 利用字符串切片筛选日期范围（包含起止日期）
            period_data = data.loc[f'{year}-{START_DATE}':f'{year}-{END_DATE}']
            
            if period_data.empty:
                continue
                
            plt.figure(figsize=(30, 20))
            plt.plot(period_data.index, period_data['COT_1_mean'])
            plt.xlabel('Time')
            plt.ylabel('PrHX_eff')
            plt.title(f'{craft_no} - {year} (Apr-Aug)')  # 添加标题区分
            plt.grid(True)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'{craft_no}_{year}_AprAug.png')
            plt.savefig(save_path, dpi=300)  # dpi=300 保证图片清晰度
            plt.close()