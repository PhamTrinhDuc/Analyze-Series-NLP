from glob import glob
import pandas as pd


def load_subtiles_dataset(dataset_path: str) -> pd.DataFrame:
    
    """
    - Đọc từng tập trong folder, mỗi tập lấy các dòng chứa hội thoại và mỗi dòng lấy ra đoạn hội thoại. 
      Cuối cùng là join các đoạn hội thoại lại thành 1 scripts cho 1 tập
    - Lấy ra sô tập

    Args:
        - đường dẫn đến dataset
    Returns:
        dataframe chứa 2 cột là script và số tập
    """

    subtiles_paths = glob(dataset_path + "/*.ass")

    scripts = []
    episodes = []

    for file_path in subtiles_paths:
        # file_path = "data/subtitlist/Naruto Season 1 - 01.ass"
        with open(file=file_path, mode='r') as file:
            lines = file.readlines()

            lines = lines[27:]
            lines = [','.join(line.split(',')[9:].strip()) for line in lines]
        
        lines = [line.replace('\N', ' ') for line in lines]

        script = " ".join(lines)
        season = file_path.split('/')[-1].split('-')[0].split("Naruto")[1].strip()
        episode_num = file_path.split('-')[-1].split('.')[0].strip()
        episode = season + " - Episode " + episode_num

        scripts.append(script)
        episodes.append(episode)

    df = pd.DataFrame.from_dict({"scripts": scripts, "episodes": episodes})
    df.to_csv("data/theme_classification.csv")
    return df