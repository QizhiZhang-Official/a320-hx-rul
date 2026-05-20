from src.data.pipeline import *
from src.data.plot import *

# process_one_craft(craft_no='B-1855', phase=2, save_dir=r'D:/zipped_data/phase_2/')

# step_1(phase=2, save_dir=r'D:/zipped_data/phase_2/')

# 画PrHX的所有图片
# plot_PrHX_eff(zipped_data_dir="D:/zipped_data/phase_6/", save_dir="figures/PrHX_analysis/phase_6/")

# 画COT的所有图片
# plot_COT(zipped_data_dir="D:/zipped_data/phase_2/", save_dir="figures/COT_analysis/phase_2/")
# plot_COT(zipped_data_dir="D:/zipped_data/phase_6/", save_dir="figures/COT_analysis/phase_6/")

# 画COT的片段 用于提取衰退曲线做分析
plot_COT_adv(
    zipped_data_dir="D:/zipped_data/phase_2/",
    craft_no="B-1816",
    start_date="2023-04-22",
    end_date="2023-05-18",
    pack=1,
    modify=1,
    is_show=True,
    save_dir="",
)

# 画所有提取的COT衰退曲线并保存
# plot_all_extracted_COT(
#     zipped_data_dir="D:/zipped_data/phase_2/", save_dir="figures/all_extracted_COT/"
# )

# 生成 annotations.yaml
# generate_annotations(
#     raw_data_dir="D:/raw_data/",
#     zipped_data_dir="D:/zipped_data/phase_2/",
#     PF_threshold=0.4,
#     max_rul_s=150 * 3600,
# )

# 依据提取的 annotations.yaml 画 RUL vs Flight Cycle 的曲线
# plot_rul_vs_FC(save_dir="figures/rul_vs_flight_cycle/")
