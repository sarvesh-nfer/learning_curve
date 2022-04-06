import sqlite3
import pandas as pd
import os,glob

txt = glob.glob("/datadrive/wsi_data/scan_data/19dec_04R/*/*.db")
df5 = pd.DataFrame()
for i in txt:
    conn = sqlite3.connect(i)
    df1= pd.read_sql_query("select * from aoi;",conn)
    df2= pd.read_sql_query("select * from focus_sampling_info;",conn)
    
    df3 = pd.merge(df2,df1,on=['aoi_name','grid_id'])
    
    df = pd.DataFrame([{'slide_name':i.split('/')[-2], 'total_pts':len(df3),'sampled_pts':len(df3[df3['is_sampled'] == 1]),
                        'removed_pts':len(df3[df3['is_sampled'] == 0]),
                       'class1_sampled_pts':len(df3[(df3['aoi_class'] == 4) & (df3['is_sampled'] == 1)]),
                        'class2_sampled_pts':len(df3[(df3['aoi_class'] == 3) & (df3['is_sampled'] == 1)]),
                       'class3_sampled_pts':len(df3[(df3['aoi_class'] == 2) & (df3['is_sampled'] == 1)]),
                        'class4_sampled_pts':len(df3[(df3['aoi_class'] == 1) & (df3['is_sampled'] == 1)]),
                       'class1_removed_pts':len(df3[(df3['aoi_class'] == 4) & (df3['is_sampled'] == 0)]),
                       'class2_removed_pts':len(df3[(df3['aoi_class'] == 3) & (df3['is_sampled'] == 0)]),
                       'class3_removed_pts':len(df3[(df3['aoi_class'] == 2) & (df3['is_sampled'] == 0)]),
                       'class4_removed_pts':len(df3[(df3['aoi_class'] == 1) & (df3['is_sampled'] == 0)])}])
    df5 = df5.append(df)

df5['higher_class_sampled'] = df5['class1_sampled_pts'] + df5['class2_sampled_pts']
df5['lower_class_sampled'] = df5['class3_sampled_pts'] + df5['class4_sampled_pts']
df5['higher_class_removed'] = df5['class1_removed_pts'] + df5['class2_removed_pts']
df5['lower_class_removed'] = df5['class3_removed_pts'] + df5['class4_removed_pts']
df5.to_csv("/home/adminspin/Music/UnitTests/sarvesh/opti_fs_19_4r.csv",index=False)
