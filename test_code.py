
file_path = "data/subtitlist/Naruto Season 1 - 01.ass"
with open(file=file_path, mode='r') as file:
    lines = file.readlines()

    lines = lines[27:]
    lines = [','.join(line.split(',')[9:]).strip() for line in lines]

lines = [line.replace("\\N",  " ") for line in lines]
scripts = ' '.join(lines)

season = file_path.split('/')[-1].split('-')[0].split("Naruto")[1].strip()
episode_num = file_path.split('-')[-1].split('.')[0].strip()

episode = season + " - Episode " + episode_num
print(episode)