from src.data.pipeline import step_1, process_one_craft
from src.data.plot import plot_PrHX_eff

# process_one_craft(craft_no='B-1855', phase=2, save_dir=r'D:/zipped_data/phase_2/')

step_1(phase=2, save_dir=r'D:/zipped_data/phase_2/')

# plot_PrHX_eff(zipped_data_dir="D:/zipped_data/phase_6/", save_dir="figures/PrHX_analysis/phase_6/")
