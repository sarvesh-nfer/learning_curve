#sarvesh

#import matplotlib.pyplot as plt
import glob
# from bokeh.plotting import figure, output_file, show
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.DataFrame()

filepath = "/datadrive/wsi_data/compressed_data/*/*.log"
txt = glob.glob(filepath)

area = []
stain = []
distance = []
sname = []
for textfile in txt:
	with open(textfile) as file:
		for line in file:
			if 'Area occupying: ' in line:
				bgTime = float(line.split(':')[-1].strip())		
				area.append(bgTime)
			if 'Acquisition started for: ' in line:
				bgTime = str(line.split(':')[-1].strip())		
				sname.append(bgTime)
			if 'Distance transform max value: ' in line:
				bgTime = float(line.split(':')[-1].strip())		
				distance.append(bgTime)
			if 'Stain color: ' in line:
				bgTime = str(line.split(':')[-1].strip())		
				stain.append(bgTime)
				

print('scanner name',len(sname))
print('Area occupying ', len(area))
print('Stain color ', len(stain))
print('Distance',len(distance))
