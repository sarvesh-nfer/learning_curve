import sys
import os
import cv2
import getopt
import re
import shutil
import string
import zipfile
import numpy as np
from time import gmtime, localtime, strftime
from datetime import datetime
import matplotlib.pyplot as plt
#import pdfkit
import pandas as pd

# df = pd.read_csv("/home/adminspin/Desktop/df.csv")
################################################################################
#Function: InitializeHTML
################################################################################
def InitializeHTML(tag):
  tag = "<!DOCTYPE html>\n"
  tag = tag + "<html>\n"
  tag = tag + "<head>\n"
  tag = tag + "<style>\n"
  tag = tag + "ul#menu {padding: 0;}\n"
  tag = tag + "ul#menu li a {background-color: white;color: black;padding: 10px 20px;text-decoration: none;}\n"
  tag = tag + "ul#menu li a:hover {background-color: orange;}\n"
  tag = tag + "</style>\n"
  tag = tag + "</head>\n"
  # tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">CMYK Profiles:"
  tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">Detection of Debris:"
  return tag

################################################################################
#Function: FinalizeHTML
################################################################################
def FinalizeHTML(header, r_pass, r_fail, table, htmlFile):
  #Append the results to the the header

  header = header + "<ul id=\"menu\">\n"
  header = header + "<table style=\"width:100%\" border=\"1\">\n"
  #Append the table 
  header = header + table 
  #Close the html 
  header = header +"</table>\n"
  header = header +"</html>"
  
  #Save the html to a file 
  f = open(htmlFile, 'w')
  f.write(header)
  f.close()
  
################################################################################
#Function: PopulateResultTable
################################################################################
def PopulateResultTable(input_folder_path, output_path,tableTag):
  tableTag = tableTag + "<th>SlideName</th><th>values</th><th>ValidInvalid</th><th>finalMergedBbox</th><th>compositeImage_whole_slide</th><th>fgmask</th>"
  # tableTag = tableTag + "<th>Input</th><th>ThresholdMask</th> <th>CMYK Profiles</th>"#<th>AnnotationRemoved</th>

  input_list = os.listdir(input_folder_path)
  # input_list = sorted(input_list)
  for item in input_list:
    try:
      # if item == "2001V401001_206" : 
      slide_path = os.path.join(input_folder_path, item)
      # blobs_list = os.listdir(os.path.join(slide_path,"loc_output_data/Blobs_cmyk"))
      # blobs_list = sorted(blobs_list)
      # for blob in blobs_list:
        # slideFolder = os.path.join(input_folder_path, item)
        # name = item.split(".")[0]
        #img1 = cv2.imread(input_folder_path+name+".bmp")
        #img2 = cv2.imread(output_path+name+".png")
      path1 = slide_path+"/loc_output_data/validInvalidGrids.jpeg"
      path2 = slide_path+"/loc_output_data/finalMergedBbox.jpeg"
      path3 = slide_path+"/loc_output_data/compositeImage_whole_slide.jpeg"
      path4 = slide_path+"/loc_output_data/fgMaskWithPoints.jpeg"
      
      # path4 = output_path+"/"+item+"/loc_output_data/AnnotationOutput_changed.png"
      print(item)
      print("path1: ",path1)
      print("path2: ",path2)
      print("path3: ",path3)
      # print("path4: ",path4)
      tableTag = tableTag + "<tr>\n"
      tableTag = tableTag +  "<th>"+str(item)+"</th>""<td>"+"<B><p>Area : </p></B>"+str(df[df['scanner_name']==item].reset_index().loc[0,"area_occupying"])+"<B><p>Distance Transform: </p></B>"+str(df[df['scanner_name']==item].reset_index().loc[0,"Distance"])
      if not os.path.exists(path1): print("Path not found : ",path1)
      if not os.path.exists(path2): print("Path not found : ",path2)
      if not os.path.exists(path3): print("Path not found : ",path3)
      if not os.path.exists(path4): print("Path not found : ",path4)

      #Add the best focused image 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path1+"\" alt=\"File Not Found\" style=\"width:45%;height:95;border:solid\"></td>\n"

      #Add the metric plot
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag + path2+"\" alt=\"File Not Found\" style=\"width:45%;height:95;border:solid\"></td>\n"

      #Add the best focused image 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path3+"\" alt=\"File Not Found\" style=\"width:45%;height:95;border:solid\"></td>\n"
      
      #Add the best focused image 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path4+"\" alt=\"File Not Found\" style=\"width:65%;height:95;border:solid\"></td>\n"

      tableTag = tableTag + "<tr>\n"
      tableTag = tableTag + "<th>SlideName</th><th>Values</th><th>ValidInvalid</th><th>finalMergedBbox</th><th>compositeImage_whole_slide</th><th>fgmask</th>"

      # tableTag = tableTag + "<tr>\n"
      # tableTag = tableTag + "<th>Input</th><th>ThresholdMask</th> <th>CMYK Profiles</th>"#<th>AnnotationRemoved</th>

      #Add the best focused image 
      # tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      # tableTag = tableTag +path4+"\" alt=\"File Not Found\" style=\"width:50%;height:90%;border:solid\"></td>\n"

      cv2.waitKey(50) 
     
    except Exception as msg:
      print("----error-----",msg)
  return tableTag



   
################################################################################
#Function: main()
#This is the main driver function for running the batch script on the data. It
#is used to branch off into multiple processes depending on what we seek to do.
#For example, if we want to rearrange the data, we branch off into the 
#Rerrange data process, else we use the regular autotest process.
################################################################################
def main():
  # pdfkit.from_file('/home/adminspin/Slides_data/401V_401/summary/summary.htm', 'out.pdf')
  # exit()
  # parse command line options
  try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
  except getopt.error as msg:
    print(msg)
    print( "for help use --help")
    sys.exit(2)
  # process options
  for o, a in opts:
    if o in ("-h", "--help"):
      #print __doc__
      print( "python3  <path_to>/BlurMetricBatch.py <path_to>/Input_Dir_Containing_Stacks <path_to>/Output_Dir_Containing_Results <path_to>/Parameter.XML <path_to_Summary_directory> <path_to_executable>/ut_spinBlurMetricStack [Optional] <0:Available,1:To_Be_Created> <path_to_GT_file.txt>")
      sys.exit(0)
  #process arguments
  filename = 'a'
  inputFolder                        = args[0]
  outputDir                          = args[1]
  summaryDir                         = args[2]
  filename                           = args[3]
  # execFile                           = args[4]
  # gtFileCreate=0
  # gtFile = ''
  dict=[]
  #if (not (len(args) == 5)):
    #print ("Give args as \n\t 1.inputDir \n\t 2.outputDir\n\t 3.white_path\n\t 4.summaryDir\n\t 5.executable_Path")

  #Check if the dummay directory exits
  if not os.path.exists(summaryDir):
    os.mkdir(summaryDir)
  if not filename == 'a':
    summaryFileName = os.path.join(summaryDir, filename + ".html")
  else:
    summaryFileName = os.path.join(summaryDir,"summary_varianceResults.html")
  r_pass = 0
  r_fail = 0
  htmlHeader = ""
  htmltable = ""
  #Initialize the header
  htmlHeader = InitializeHTML(htmlHeader)
  
 
  htmltable = PopulateResultTable(inputFolder, outputDir, htmltable)
  FinalizeHTML(htmlHeader, r_pass, r_fail, htmltable, summaryFileName)
  exit()
  
if __name__ == "__main__":
    main()
