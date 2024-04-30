import os
import pandas as pd

def group_solar_columns(dataframe):
    solar_columns = [col for col in dataframe.columns if 'solar' in col]
    grouped_data = []

    for index, row in dataframe.iterrows():
        num_solar_columns = sum(1 for col in solar_columns if not pd.isnull(row[col]))

        if num_solar_columns == 4:
            grouped_data.append([row[f'solar_{i}'] for i in range(1, 5)])
        elif num_solar_columns % 4 == 0:
            group_size = num_solar_columns // 4
            group = []
            for i in range(group_size):
                group.append([row[f'solar_{j}'] for j in range(i + 1, num_solar_columns + 1, group_size)])
            grouped_data.append(group)
        else:
            remainder = num_solar_columns % 4
            group_size = num_solar_columns // 4
            group = []
            start_col = 1
            for i in range(group_size):
                end_col = start_col + remainder - 1 if i < remainder else start_col + remainder
                group.append([row[f'solar_{j}'] for j in range(start_col, end_col + 1)])
                start_col = end_col + 1
            grouped_data.append(group)

    return grouped_data


def create_nested_label_list(dataframe):
    grouped_data = group_solar_columns(dataframe)
    dataframe['coordinate'] = [[] for _ in range(len(dataframe))]

    for i, group in enumerate(grouped_data):
        if isinstance(group[0], list):
            for sublist in group:
                cleaned_sublist = [val for val in sublist if pd.notnull(val)]
                if cleaned_sublist:
                    dataframe.at[i, 'coordinate'].append(cleaned_sublist)
        else:
            cleaned_group = [val for val in group if pd.notnull(val)]
            if cleaned_group:
                dataframe.at[i, 'coordinate'].append(cleaned_group)

    return dataframe

def main():
    directory = "./label_data/"
    filenames = [filename for filename in os.listdir(directory) if filename.endswith(".csv")]

    files = []
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        file = pd.read_csv(file_path)
        file['files'] = file['files'].str.replace(r'D:\\', r'Z:\\Projects\\202404_태양광\\', regex=True)
        file_with_labels = create_nested_label_list(file)
        files.append(file_with_labels)

    df = pd.concat(files, axis=0, ignore_index=True)
    df = df[['files', 'coordinate']]

    output_dir = './output/preprocess_label'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(os.path.join(output_dir, 'preprocessed_label.csv'), index=False, encoding='utf-8-sig')
    print(df)

if __name__ == '__main__':
    main()
