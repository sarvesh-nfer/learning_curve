import os, sys
import pandas as pd

def log_parser(input_file_path):
    df = pd.DataFrame()
    df_all = pd.DataFrame()
    camera_board = []
    camera_sensor = []
    files_read = os.listdir(input_file_path)

    for idx in files_read:
        print(idx)
        txt = os.path.join(input_file_path, idx )

        with open(txt) as file:
            print(txt)
            data_to_parse = []
            for line in file:
                try:
                    data_to_parse.append(line)
                    for line in data_to_parse:
                        if 'Acquisition started for' in line:
                            name = line.split(" ")[-2].replace('\n','')
                        if 'Micro Imaging Camera Board Temperature' in line:
                            board = line.split(" ")[-1].replace('\n','')
                        if 'Micro Imaging Camera Sensor Temperature' in line:
                            sensor = line.split(" ")[-1].replace('\n','')

                            camera_board.append(board)
                            camera_sensor.append(sensor)
                        data_to_parse = []
                except Exception as msg:
                    print(msg)

        df = pd.DataFrame(list(zip(camera_board,camera_sensor)),columns =['camera_board','camera_sensor'])
        df['slide_name'] = name

        df_all = df_all.append(df)

    return df_all

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]  
        try:
            df = log_parser(input_file_path)
            df.to_csv(output_file_path + "/log_parsed.csv",index=False)

            df1 = pd.read_csv(output_file_path + "/log_parsed.csv")
            df1 = df1.groupby(["slide_name"])[["camera_board","camera_sensor"]].median().reset_index()
            df1.to_csv(output_file_path + "/log_parsed_group.csv",index=False)
            print(output_file_path + "/log_parsed_group.csv")
        except Exception as msg:
            print(msg)
