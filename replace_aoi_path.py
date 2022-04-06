import pandas as pd
import os,glob
import sys

def generateShiftExcel(path):
    acq = os.path.join(path+"/grid_1/AcquisitionData.xlsx")
    shift = os.path.join(path+"/grid_1/acquisition_debug_data/stack_images/Shift.xlsx")

    df1 = pd.read_excel(acq)
    df2 = pd.read_excel(shift)

    for i,j,k in zip(df2['AOI Name'],df2['Maximum X Shift'],df2['Maximum Y Shift']):
        xshift = df1.replace(to_replace =str(i),value =str(j))
        yshift = df1.replace(to_replace =str(i),value =str(k))
        

    xshift.to_excel(path+"/grid_1/xshift.xlsx",index=False)
    yshift.to_excel(path+"/grid_1/yshift.xlsx",index=False)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    generateShiftExcel(slide_path)
