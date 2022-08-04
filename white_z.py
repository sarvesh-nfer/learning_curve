import glob
import pandas as pd

sar = {}

for i in df2['slide_name'].unique():
    try:
        conn = sqlite3.connect("/home/adminspin/Desktop/data_test"+"/"+i+"/"+i+".db")
        
        df = pd.read_sql_query("select * from aoi",conn)
        print(df[df['aoi_name'] == df2[df2['slide_name']==i].iloc[0][1]].iloc[0]['best_z'])
        
        sar[i] = df[df['aoi_name'] == df2[df2['slide_name']==i].iloc[0][1]].iloc[0]['best_z']
        
        
    except:
        print("error in ",i)

data = pd.DataFrame(sar.items(),columns=['slide_name','best_z'])
data.to_csv("/home/adminspin/Music/sar.csv",index=False)
