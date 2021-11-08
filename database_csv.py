import tarfile
import glob,os,sys
import sqlite3
import pandas as pd

def extract_other(slide_path):
    txt = glob.glob(slide_path+"/*/other.tar")
    for i in txt:
        try:
            my_tar = tarfile.open(i)
            my_tar.extractall(os.path.split(i)[0])
            my_tar.close()
            print("**"*50)
            print("Slide Successfully Extracted : ",i.split('/')[-2])
            print("**"*50)
        except Exception as msg:
            print("--"*50)
            print("Couldn't Extract Slide : ",i.split('/')[-2],"/t msg :",msg)
            print("--"*50)


def get_log_stuff(slide_path):
    df = pd.DataFrame()
    class_H_and_E = []
    stain_color = []
    biopsy_type = []
    slide_name = []
    lst_grid_ext = []
    lst_g_grid_id = []
    lst_opt_fs = []
    lst_fs_grid_id = []

    txt = glob.glob(slide_path+"/*/*.log")

    for textfile in txt:
        try:
            with open(textfile) as file:
                data_to_parse = []
                for line in file:
                    data_to_parse.append(line)
                for line in data_to_parse:
                    if 'Stain classification using highest probability:' in line:
                        type_ = line.split(':')[-1].strip()
                    if 'Stain color:' in line:
                        stain_ = line.split(':')[-1].strip()
                    if 'Biopsy type' in line:
                        b_type_ = line.split(':')[-1].strip()
                    if 'grid_extrapolation_from_plane_status:' in line:
                        grid_ext = line.split(',')[-2].split(':')[-1].strip()
                        g_grid_id = line.split(',')[-1].split(':')[-1].strip()
                    if 'optimised_fs_from_plane_status:' in line:
                        opt_fs = line.split(',')[-2].split(':')[-1].strip()
                        fs_grid_id = line.split(',')[-1].split(':')[-1].strip() 
                        
                        lst_grid_ext.append(grid_ext)
                        lst_g_grid_id.append(g_grid_id)
                        lst_opt_fs.append(opt_fs)
                        lst_fs_grid_id.append(fs_grid_id)
                        slide_name.append(textfile.split('/')[-2].strip())
                        class_H_and_E.append(type_)
                        stain_color.append(stain_)
                        biopsy_type.append(b_type_)
        except Exception as msg:
            print("Error in slide : ",textfile.split('/')[-2],"\t msg : ",msg)
    
    df = pd.DataFrame(list(zip(slide_name,lst_grid_ext,lst_g_grid_id,lst_opt_fs,lst_fs_grid_id,class_H_and_E,stain_color,biopsy_type)),
    columns =['slide_name','grid_ext','g_grid_id','opt_fs','fs_grid_id','class_H&E','stain_color','biopsy_type'])

    dfa = df[['slide_name','grid_ext']].value_counts().reset_index(name='counts').pivot_table('counts', ['slide_name'], 'grid_ext').reset_index()
    dfa.rename(columns={'False': 'grid_ext_false', 'True': 'grid_ext_true'}, inplace=True)
    dfa['grid_ext_false'].fillna(0,inplace=True)
    dfa['grid_ext_true'].fillna(0,inplace=True)
    dfa[['grid_ext_false','grid_ext_true']] =dfa[['grid_ext_false','grid_ext_true']].astype(int)

    dfb = df[['slide_name','opt_fs']].value_counts().reset_index(name='counts').pivot_table('counts', ['slide_name'], 'opt_fs').reset_index()
    dfb.rename(columns={'False': 'opt_fs_false', 'True': 'opt_fs_true'}, inplace=True)
    dfb['opt_fs_false'].fillna(0,inplace=True)
    dfb['opt_fs_true'].fillna(0,inplace=True)
    dfb[['opt_fs_false','opt_fs_true']] = dfb[['opt_fs_false','opt_fs_true']].astype(int)

    df_f = df[['slide_name','class_H&E','stain_color','biopsy_type']].drop_duplicates(subset=['slide_name'],keep='last')

    log_output = df_f.merge(dfa,on='slide_name').merge(dfb,on='slide_name')
    return log_output


def database_csv(slide_path):
    txt = glob.glob(slide_path+"/*/*.db")
    df_final = pd.DataFrame()
    # df = pd.DataFrame()
    for i in txt:
        try:
            conn = sqlite3.connect(i)
            cur = conn.cursor()
            df_focus = pd.read_sql_query("select * from focus_sampling_info ;", conn)
            df_val = pd.read_sql_query("select * from validation_info ;", conn)
            plane = pd.read_sql_query("select * from grid_info ;", conn)

            # inline_plane = len(plane[plane['plane_status'] == 1])

            df_focus = df_focus[(df_focus['focus_metric'] >= 0) & (df_focus['color_metric'] >= 0)]

            # print("BZ_valid_time : ",sum(df_val[df_val['best_z'] > 0]['process_time']))

            df = pd.DataFrame([[i.split('/')[-2],len(df_val),len(df_val[df_val['best_z'] > 0]),len(df_val[df_val['best_z'] < 0]),
            round(sum(df_val[df_val['best_z'] > 0]['process_time']),2),round(sum(df_val[df_val['best_z'] < 0]['process_time']),2),round(sum(df_val['process_time']),2),
            len(df_focus),len(df_focus[df_focus['is_sampled'] == 0]),len(df_focus[df_focus['status'] == 1]),len(df_focus[df_focus['status'] == 0]),round(sum(df_focus[df_focus['status'] == 1]['process_time']),2),
            round(sum(df_focus[df_focus['status'] == 0]['process_time']),2),round(sum(df_focus['process_time']),2),len(plane),len(plane[plane['plane_status'] == 1])]],
            columns=['slide_name','BZ_total_points','BZ_valid','BZ_invalid','BZ_valid_time(s)','BZ_invalid_time(s)','BZ_total_time(s)',
            'FS_total_points','FS_removed_points','FS_valid','FS_invalid','FS_valid_time(s)','FS_invalid_time(s)','FS_total_time(s)','total_grids','inline_plane_grids'])

            df_final = df_final.append(df)
            df_final.to_excel(slide_path+'/whole_slide_data.xlsx',index=False)
            print("&&"*50)
            print("Slide Successfully Saves : ",i.split('/')[-2])
            print("&&"*50) 
            # print(df_final)
        except Exception as msg:
            print("--"*50)
            print("Couldn't read DB for Slide : ",i.split('/')[-2],"/t msg :",msg)
            print("--"*50)
    log_stuff = get_log_stuff(slide_path)
    log_merged = pd.merge(df_final,log_stuff,on='slide_name')
    
    log_merged.to_excel("/home/adminspin/Desktop/log_merged.xlsx",index=False)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    print(slide_path)
    extract_other(slide_path)
    print("**"*50)
    print("Started to save CSV")
    database_csv(slide_path)