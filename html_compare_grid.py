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
import sqlite3
import pandas as pd
# import pdfkit

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
  tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">Best-Z Experiment "
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
  # tableTag = tableTag + "<th>Input</th><th>ThresholdMask</th> <th>CMYK Profiles</th>"#<th>AnnotationRemoved</th>

  #loc folder
  input_list = os.listdir(input_folder_path)
  input_list = sorted(input_list, reverse = True)
  input_list = sorted(input_list)
  count = 0; max_count = 200
  for item in input_list:
    count +=1
    if(count>max_count):
      break
    print(count)
    slide_path = os.path.join(input_folder_path, item)
    print(slide_path)
    try:
      path1 = slide_path+"/" + "loc_output_data" + "/whiteCorrectedInput.png"
      path2 = slide_path+"/" + "loc_output_data" + "/best_z_order.png"
      path3 = slide_path+"/" + "loc_output_data" + "/possible_annotations.jpeg"
      path4 = slide_path+"/" + "loc_output_data" + "/compositeImage.jpeg"
      path5 = slide_path+"/" + "loc_output_data" + "/colored_class_info.png"
      # db_path = os.path.join(slide_path,item+".db")
      
      # conn = sqlite3.connect(db_path)
      # df = pd.read_sql_query("select * from validation_info;",conn)


      print("path1: ",path1)
      print("path2: ",path2)
      print("path3: ",path3)
      print("path4: ",path4)

      tableTag = tableTag + "<tr>\n"
      tableTag = tableTag + "<th>"+item+"</th><th>best_z_order</th><th>possible_annotations</th><th>compositeImage</th><th>colored_class_info</th>"
      tableTag = tableTag + "<tr>\n"
                                                                                
      if not os.path.exists(path1): print("Path not found : ",path1)
      if not os.path.exists(path2): print("Path not found : ",path2)
      if not os.path.exists(path3): print("Path not found : ",path3)
      if not os.path.exists(path4): print("Path not found : ",path4)
      # if not os.path.exists(path4): print("Path not found : ",path6)

      #Add the best focused image 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path1+"\" alt=\"File Not Found\" style=\"width:65%;height:120;border:solid\"></td>\n"

      #Add the ORIGINAL mask
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag + path2+"\" alt=\"File Not Found\" style=\"width:65%;height:120;border:solid\"></td>\n"

      #Add the Model MASK 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path3+"\" alt=\"File Not Found\" style=\"width:65%;height:50;border:solid\"></td>\n"

      #Add the Aug MASK 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path4+"\" alt=\"File Not Found\" style=\"width:65%;height:95;border:solid\"></td>\n"

      #Add the Aug MASK 
      tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
      tableTag = tableTag +path5+"\" alt=\"File Not Found\" style=\"width:65%;height:95;border:solid\"></td>\n"


      cv2.waitKey(50) 
    except Exception as msg:
      print("----error-----", "[error]: ", msg)
  return tableTag


################################################################################
#Function: PopulateGT
################################################################################
def PopulateGT(fName):
  f = open(fName, 'r')
  dict={}
  
  #Get the matching pairs
  for line in f.readlines():
      line = line.strip()
      columns = line.split()
      dict[str(columns[0])] = str(columns[1])

  f.close()
  return dict
  
################################################################################
#Function: checkKey
################################################################################
def checkKey(dict, key):   
  val=''    
  if key in dict:
    val = dict[key] 
  return val
    
################################################################################
#Function: plotMetric Use this function when a GT file exists in the disk and 
#          we need to get a count of pass/fail
################################################################################
def plotMetric(fName, fPlot, bFPath, stackName, gt, r_pass, r_fail, tableTag):
  f = open(fName, 'r')
  listX=[]
  listY=[]
  numC = 0
  plt.grid(True)
  
  #Get the list of images
  for line in f.readlines():
      line = line.strip()
      columns = line.split()
      numC = len(columns)
      a=columns[0]
      listX.append(a)
  f.close()
  
  #Plot all curves simultaneously
  for idx in range(1,numC):
    f = open(fName, 'r')
    listY=[]      
    for line in f.readlines():
      line = line.strip()
      columns = line.split()
      b=columns[idx]
      listY.append(float(b))
  
    plt.plot(listX, listY,'-*')
    f.close()

  #Scan the Best focus folder 
  bestFocus = os.listdir(bFPath)
  pred=''
  if (len(bestFocus) == 1):
    pred = bestFocus[0]
  
  plt.xlabel('Image Name')
  plt.ylabel('Metric Value')
  
  check = 0
  if (pred==gt):
    gtX = stackName + ': '+pred+' <--> '+gt
    plt.title(gtX, fontsize=20, color='g')
    r_pass = r_pass + 1
    check = 1
  else:
    gtX = stackName + ': '+pred+' <--> '+gt 
    plt.title(gtX, fontsize=20, color='r')
    r_fail = r_fail + 1
  
  bFImage = os.path.join(bFPath,pred)
  tableTag = PopulateResultTable(fPlot, bFImage, check, stackName, tableTag)
    
  fig = plt.gcf()
  fig.set_size_inches(10,8,forward=True)
  fig.savefig(fPlot,dpi=100)
  fig.clf()

  return r_pass, r_fail, tableTag

################################################################################
#Function: plotMetric_NoGT Use this function when we do not have a GT and want
#          to create a GT.
################################################################################
def plotMetric_NoGT(fName, fPlot, bFPath, stackName, gtFile, tableTag):
  f = open(fName, 'r')
  listX=[]
  listY=[]
  numC = 0
  plt.grid(True)
  
  #Get the list of images
  for line in f.readlines():
      line = line.strip()
      columns = line.split()
      numC = len(columns)
      a=columns[0]
      listX.append(a)
  f.close()
  
  #Plot all curves simultaneously
  for idx in range(1,numC):
    f = open(fName, 'r')
    listY=[]      
    for line in f.readlines():
      line = line.strip()
      columns = line.split()
      b=columns[idx]
      listY.append(float(b))
  
    plt.plot(listX, listY,'-*')
    f.close()

  #Scan the Best focus folder 
  bestFocus = os.listdir(bFPath)
  pred=''
  if (len(bestFocus) == 1):
    pred = bestFocus[0]
  
  plt.xlabel('Image Name')
  plt.ylabel('Metric Value')
  
  check = 0
  gtX = stackName + ': '+pred+' <--> ' 
  plt.title(gtX, fontsize=20, color='r')
  
  bFImage = os.path.join(bFPath,pred)
  tableTag = PopulateResultTable(fPlot, bFImage, check, stackName, tableTag)
    
  fig = plt.gcf()
  fig.set_size_inches(10,8,forward=True)
  fig.savefig(fPlot,dpi=100)
  fig.clf()
  
  #Populate the GT file 
  f = open(gtFile,'a+')
  s = stackName+' '+pred+'\n'
  f.write(s)
  f.close()

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
  '''
  if (len(args) > 5):
    gtFileCreate                     = int(args[5])
    gtFile                           = args[6]
    #Populate GT information (if gtFileCreate is 0)
    if (gtFileCreate == 0):
      dict = PopulateGT(gtFile)
    else:
      #We need to create a GT File and keep on appending information to that
      f = open(gtFile,'w')
      f.close()
  '''
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
  
  '''# changed part
  # cLine = [execFile,'"'+inputFolder+'"','"'+outputDir+'"','"'+white_path+'"'] 
    #http://anh.cs.luc.edu/python/hands-on/3.1/handsonHtml/strings3.html
  cmdLine = ' '.join(cLine)
  print( cmdLine)
  print( "===========================\n")
  os.system(cmdLine)
  '''
 
  htmltable = PopulateResultTable(inputFolder, outputDir, htmltable)
  FinalizeHTML(htmlHeader, r_pass, r_fail, htmltable, summaryFileName)
  exit()
  
if __name__ == "__main__":
    main()
