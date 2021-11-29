import tarfile
import glob,os,sys
import sqlite3
import pandas as pd
import plotly.express as px
from html_compare_grid import html_generation

def extract_other(slide_path):
    txt = glob.glob(slide_path+"/*/other.tar")
    count = 1
    for i in txt:
        try:
            my_tar = tarfile.open(i)
            my_tar.extractall(os.path.split(i)[0])
            my_tar.close()
            print("**"*50)
            print("Slide Successfully Extracted : ",i.split('/')[-2],"\t COUNT :",count)
            count +=1
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
    # log_time = []
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
                    # if 'Total time acquire slide' in line:
                    #     log_time.append(line.split(':')[-1].strip())
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
                        # log_time.append(l_time)
        except Exception as msg:
            print("Error in slide : ",textfile.split('/')[-2],"\t msg : ",msg)
    
    df = pd.DataFrame(list(zip(slide_name,lst_grid_ext,lst_g_grid_id,lst_opt_fs,lst_fs_grid_id,class_H_and_E,stain_color,biopsy_type)),
    columns =['slide_name','grid_ext','g_grid_id','opt_fs','fs_grid_id','class_H&E','stain_color','biopsy_type'])

    dfa = df[['slide_name','grid_ext']].value_counts().reset_index(name='counts').pivot_table('counts', ['slide_name'], 'grid_ext').reset_index()
    dfa.rename(columns={'False': 'grid_ext_false', 'True': 'grid_ext_true'}, inplace=True)
    # print(dfa)
    dfa['grid_ext_false'].fillna(0,inplace=True)
    dfa['grid_ext_true'].fillna(0,inplace=True)
    dfa[['grid_ext_false','grid_ext_true']] =dfa[['grid_ext_false','grid_ext_true']].astype(int)

    dfb = df[['slide_name','opt_fs']].value_counts().reset_index(name='counts').pivot_table('counts', ['slide_name'], 'opt_fs').reset_index()
    dfb.rename(columns={'False': 'opt_fs_false', 'True': 'opt_fs_true'}, inplace=True)
    # print(dfb)
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
    count = 1
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
            round(sum(df_val[df_val['best_z'] > 0]['process_time']),2),round(sum(df_val[df_val['best_z'] < 0]['process_time']),2),
            round(sum(df_val['process_time']),2),len(df_focus),len(df_focus[df_focus['is_sampled'] == 0]),
            len(df_focus[df_focus['status'] == 1]),len(df_focus[df_focus['status'] == 0]),
            round(sum(df_focus[df_focus['status'] == 1]['process_time']),2),round(sum(df_focus[df_focus['status'] == 0]['process_time']),2),
            round(sum(df_focus['process_time']),2),len(plane),len(plane[plane['plane_status'] == 1]),sum(plane['bg_count']),
            sum(plane['fg_count']),round(sum(plane['acq_time']))]],
            columns=['slide_name','BZ_total_points','BZ_valid','BZ_invalid','BZ_valid_time(s)','BZ_invalid_time(s)','BZ_total_time(s)',
            'FS_total_points','FS_removed_points','FS_valid','FS_invalid','FS_valid_time(s)','FS_invalid_time(s)','FS_total_time(s)',
            'total_grids','inline_plane_grids','total_bg_count','total_fg_count','total_acq_time(s)'])

            df_final = df_final.append(df)
            # df_final.to_excel(slide_path+'/whole_slide_data.xlsx',index=False)
            print("&&"*50)
            print("Slide Successfully Saves : ",i.split('/')[-2],"\t COUNT :",count)
            count +=1
            print("&&"*50) 
            # print(df_final)
        except Exception as msg:
            print("--"*50)
            print("Couldn't read DB for Slide : ",i.split('/')[-2],"/t msg :",msg)
            print("--"*50)
    log_stuff = get_log_stuff(slide_path)
    log_merged = pd.merge(df_final,log_stuff,on='slide_name')
    if not os.path.exists(slide_path+"/Analysis"):
        os.mkdir(slide_path+"/Analysis")
    log_merged.to_excel(slide_path+"/Analysis/log_merged.xlsx",index=False)

    fig1= px.line(x = log_merged['slide_name'],y= log_merged['BZ_total_time(s)'],color=log_merged['biopsy_type'],markers=True)
    fig1.update_xaxes(title = "Slide Name")
    fig1.update_yaxes(title = "Best-Z time (s)")
    fig1.add_bar(x = log_merged['slide_name'],y = log_merged['BZ_total_points'],text=log_merged['BZ_total_points'],name='Total BZ Points')
    fig1.update_layout(legend_title_text='Legend',width = 2000,height = 1000,yaxis = dict(tickfont = dict(size=20)),xaxis = dict(tickfont = dict(size=20)))
    fig1.update_layout(legend=dict(x=0.01,y=0.99,traceorder="reversed",title_font_family="Times New Roman",font=dict(family="Courier",size=15,
    color="black"),bgcolor="LightSteelBlue",bordercolor="Black",borderwidth=2))
    fig1.update_layout(hovermode="x")
    # fig1.show()
    fig1.write_html(slide_path+"/Analysis/best_z.html")

    fig1= px.line(x = log_merged['slide_name'],y= log_merged['FS_total_time(s)'],color=log_merged['biopsy_type'],markers=True)
    fig1.update_xaxes(title = "Slide Name")
    fig1.update_yaxes(title = "FS time (s)")
    fig1.add_bar(x = log_merged['slide_name'],y = log_merged['FS_valid'],text=log_merged['FS_valid'],name='Valid FS Points')
    fig1.update_layout(legend_title_text='Legend',width = 2000,height = 1000,yaxis = dict(tickfont = dict(size=20)),xaxis = dict(tickfont = dict(size=20)))
    fig1.update_layout(legend=dict(x=0.01,y=0.99,traceorder="reversed",title_font_family="Times New Roman",font=dict(family="Courier",size=15,
    color="black"),bgcolor="LightSteelBlue",bordercolor="Black",borderwidth=2))
    fig1.update_layout(hovermode="x")
    # fig1.show()
    fig1.write_html(slide_path+"/Analysis/FS.html")

    fig1= px.line(x = log_merged['slide_name'],y = log_merged['total_acq_time(s)'],text=log_merged['total_acq_time(s)'],color=log_merged['biopsy_type'],markers=True)
    fig1.update_xaxes(title = "Slide Name")
    fig1.update_yaxes(title = "Total Acquitision time (s)")
    fig1.add_bar(x = log_merged['slide_name'],y = log_merged['total_bg_count'],text=log_merged['total_bg_count'],name='Total BG')
    fig1.add_bar(x = log_merged['slide_name'],y = log_merged['total_fg_count'],text=log_merged['total_fg_count'],name='Total FG')
    # fig1.update_traces(line=dict(width=5))
    fig1.update_layout(legend_title_text='Legend',width = 6000,height = 2000,yaxis = dict(tickfont = dict(size=60)),xaxis = dict(tickfont = dict(size=60)))
    fig1.update_layout(legend=dict(x=0.01,y=0.99,traceorder="reversed",title_font_family="Times New Roman",font=dict(family="Courier",size=40,
    color="black"),bgcolor="LightSteelBlue",bordercolor="Black",borderwidth=2))
    fig1.update_layout(hovermode="x")
    # fig1.show()
    fig1.write_html(slide_path+"/Analysis/total.html")



if __name__ == '__main__':
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    print(slide_path)
    # extract_other(slide_path)
    print("**"*50)
    print("Started to save CSV")
    database_csv(slide_path)
    html_name = os.path.split(slide_path)[-1]
    html_generation.main(slide_path,slide_path+"/Analysis",slide_path+"/Analysis",html_name)


