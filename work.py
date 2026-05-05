from src.data.pipeline import step_1, process_one_craft
from src.data.plot import plot_PrHX_eff, plot_COT_all, plot_COT_adv

# process_one_craft(craft_no='B-1855', phase=2, save_dir=r'D:/zipped_data/phase_2/')

step_1(phase=2, save_dir=r'D:/zipped_data/phase_2/')

# 画PrHX的所有图片
# plot_PrHX_eff(zipped_data_dir="D:/zipped_data/phase_6/", save_dir="figures/PrHX_analysis/phase_6/")

# 画COT的所有图片
# plot_COT(zipped_data_dir="D:/zipped_data/phase_2/", save_dir="figures/COT_analysis/phase_2/")
# plot_COT(zipped_data_dir="D:/zipped_data/phase_6/", save_dir="figures/COT_analysis/phase_6/")

# 画COT的片段 用于提取衰退曲线做分析
# plot_COT_adv(
#     zipped_data_dir="D:/zipped_data/phase_2/",
#     craft_no="B-1816",
#     start_time="2023-04-01",
#     end_time="2023-05-18",
#     pack=1,
#     modify=1,
# )
