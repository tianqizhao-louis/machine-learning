import pandas as pd


def process_to_pytorch(file_path, max_lines):
    tsv_file = pd.read_csv(filepath_or_buffer=file_path, delimiter='\t', quoting=3, nrows=max_lines, header=None)

    # using lowercase
    tsv_file.iloc[:, 1] = tsv_file.iloc[:, 1].str.lower()

    tsv_file.iloc[:, 0] = tsv_file.iloc[:, 0].astype(float)
    for row_number in range(len(tsv_file.iloc[:, 0])):
        if tsv_file.iloc[row_number, 0] == 2:
            tsv_file.iloc[row_number, 0] = 1.0
        else:
            tsv_file.iloc[row_number, 0] = 0.0

    return_list = []
    for row_number in tsv_file.index:
        return_list.append((tsv_file.iloc[row_number][0], tsv_file.iloc[row_number][1]))

    return return_list
