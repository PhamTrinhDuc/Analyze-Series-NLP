import pandas as pd
from configs.configurator import CONFIGURATOR



df = pd.read_csv(CONFIGURATOR.SAVE_THEME_SCRIPT_PATH)
print(df.head())