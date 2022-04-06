import glob,os
import pandas as pd
import sqlite3


txt = glob.glob("/home/adminspin/Downloads/coarse_analysis/04R_dbs/*/*.db")

df = pd.DataFrame()
df2 = pd.DataFrame()
for i in txt:
#     db_path = i
    conn = sqlite3.connect(i)
#     print("slide_name : ",db_path.split('/')[-2])
    df_aoi = pd.read_sql_query("select * from aoi ;", conn)
    df_val = pd.read_sql_query("select * from validation_info ;",conn)
    df_focus = pd.read_sql_query("select * from focus_sampling_info ;",conn)
    df_grid = pd.read_sql_query("select * from grid_info ;",conn)
    df_slide = pd.read_sql_query("select * from slide_characteristics ;",conn)
    
    df_grid['z_varition'] = abs(df_grid['plane_est_z'] - df_grid['actual_z'])
    
    df['slide_name'] = i.split('/')[-2]
    print(df['slide_name'])
    df['stain_type'] = df_slide['stain_type']

    if len(df_aoi[df_aoi['plane_estimated_z'] > 0]) > 0:
        df['plane_used_acq'] = 'True'
    else:
        df['plane_used_acq'] = 'False'

    if len(df_val[df_val['ref_z_plane'] > 0]) > 0:
        df['plane_used_val'] = 'True'
    else:
        df['plane_used_val'] = 'False'

    if len(df_focus[df_focus['is_sampled'] == 0]) > 0:
        df['removed_FS_point'] = 'True'
    else:
        df['removed_FS_point'] = 'False'

    lst = ['North','South','East','West']

    if len(df_val[df_val['direction'].isin(lst)]) >0:
        df['grid_validation'] = 'True'
    else:
        df['grid_validation'] = 'False'
    
    if len(df_grid[(df_grid['z_varition'] < 3.75) & (df_grid['plane_status'] == 1)]) > 0:
        df['grid_extrapolation'] = 'True'
    else:
        df['grid_extrapolation'] = 'False'
    df['count_grid_extrapolation'] = len(df_grid[(df_grid['z_varition'] < 3.75) & (df_grid['plane_status'] == 1)])
    
    
#     print(df)
    df2 = df2.append(df)
df2.to_csv("/home/adminspin/Desktop/grid_extra_tried.csv",index=False)

