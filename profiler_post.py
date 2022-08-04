import cv2
import numpy as np
from skimage import draw
import glob
import plotly.graph_objects as go


r_start = 0
c_start = 540
r_end = 1912
c_end = 565


fig = go.Figure()
for i in glob.glob("/hdd_drive/post_log/*/empty_slot_stack/*.bmp"):
    try:

        img = cv2.imread(i,0)
        image = cv2.line(img.copy(), (r_start, c_start), (r_end, c_end), (255, 255, 255), 2)
        line = np.transpose(np.array(draw.line(r_start, c_start, r_end, c_end)))
        data = img.copy()[line[:, 1]]
        
        fig.add_trace(go.Scatter(y=data[0],
            marker=dict(color="blue"),opacity=0.5,name="<b>Z"i.split("/")[-1].split(".")[0]))
    except:
        print(i)
fig.write_image(("/hdd_drive/post_log//profile.png")
        
        
        
        
