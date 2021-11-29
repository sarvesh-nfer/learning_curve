import sys
import os
# import cv2
import getopt
import re
import shutil
import string
import zipfile
import numpy as np
from time import gmtime, localtime, strftime
from datetime import datetime
import matplotlib.pyplot as plt
# import pdfkit

################################################################################
#Function: InitializeHTML
################################################################################

class html_generation():


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
    tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">Grid_validation "
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
        #path3 = slide_path+"/" + "loc_output_data" + "/fgMaskWithPoints.jpeg"
        path3 = slide_path+"/" + "loc_output_data" +  "/updatedImageWithBBoxMerged.jpeg"
        path4 = slide_path+"/" + "loc_output_data" + "/validInvalidGrids.jpeg"
        path5 = slide_path+"/" + "loc_output_data" + "/finalMergedBbox.jpeg"

        path6 = slide_path+"/" + "loc_output_data" + "/foregroundMask.png"

        print("path1: ",path1)
        print("path2: ",path2)
        print("path3: ",path3)
        print("path4: ",path4)
        print("path5: ",path5)
        print("path6: ",path6)
        
        if not os.path.exists(path1):
          path1 = slide_path+"/" + "loc_output_data" + "/updatedInputImage.jpeg"

        if not os.path.exists(path2):
          path2 = slide_path+"/" + "loc_output_data" + "/foregroundMask.jpeg"

        if not os.path.exists(path6):
          continue
        tableTag = tableTag + "<tr>\n"
        tableTag = tableTag + "<th>"+item+"</th><th>best_z_order</th><th>updatedImageWithBBoxMerged</th><th>validInvalidGrids</th><th>finalMergedBbox</th><th>total_foreground</th>"
        tableTag = tableTag + "<tr>\n"
                                                                                  
        if not os.path.exists(path1): print("Path not found : ",path1)
        if not os.path.exists(path2): print("Path not found : ",path2)
        if not os.path.exists(path3): print("Path not found : ",path3)
        if not os.path.exists(path4): print("Path not found : ",path4)
        if not os.path.exists(path4): print("Path not found : ",path5)
        # if not os.path.exists(path4): print("Path not found : ",path6)

        #Add the best focused image 
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag +path1+"\" alt=\"File Not Found\" style=\"width:65%;height:120;border:solid\"></td>\n"

        #Add the ORIGINAL mask
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag + path2+"\" alt=\"File Not Found\" style=\"width:65%;height:120;border:solid\"></td>\n"

        #Add the Model MASK 
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag +path3+"\" alt=\"File Not Found\" style=\"width:35%;height:50;border:solid\"></td>\n"

        #Add the Aug MASK 
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag +path4+"\" alt=\"File Not Found\" style=\"width:35%;height:95;border:solid\"></td>\n"

        #Add the Aug MASK 
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag +path5+"\" alt=\"File Not Found\" style=\"width:35%;height:95;border:solid\"></td>\n"

        #Add the Aug MASK 
        tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
        tableTag = tableTag +path6+"\" alt=\"File Not Found\" style=\"width:35%;height:95;border:solid\"></td>\n"

        cv2.waitKey(50) 
      except Exception as msg:
        print("----error-----", "[error]: ", msg)
    return tableTag
    
################################################################################
#Function: main()
#This is the main driver function for running the batch script on the data. It
#is used to branch off into multiple processes depending on what we seek to do.
#For example, if we want to rearrange the data, we branch off into the 
#Rerrange data process, else we use the regular autotest process.
################################################################################
  def main(inputFolder,outputDir,summaryDir,filename):
    # pdfkit.from_file('/home/adminspin/Slides_data/401V_401/summary/summary.htm', 'out.pdf')
    # exit()
    # parse command line options

    #process arguments
    inputFolder                        = inputFolder
    outputDir                          = outputDir
    summaryDir                         = summaryDir
    filename                           = filename
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
    htmlHeader = html_generation.InitializeHTML(htmlHeader)
  
    htmltable = html_generation.PopulateResultTable(inputFolder, outputDir, htmltable)
    html_generation.FinalizeHTML(htmlHeader, r_pass, r_fail, htmltable, summaryFileName)
    exit()
  
# if __name__ == "__main__":
    # main("/home/adminspin/Desktop/16th_nov_4R","/home/adminspin/Desktop","/home/adminspin/Desktop",'new2')
    # inputFolder = "/home/adminspin/Desktop/16th_nov_4R"
    # outputDir = "/home/adminspin/Desktop"
    # summaryFileName = os.path.join(outputDir, "localization.html")
    # r_pass = 0
    # r_fail = 0
    # htmlHeader = ""
    # htmltable = ""
    # #Initialize the header
    # htmlHeader = html_generation.InitializeHTML(htmlHeader)
  
    # htmltable = html_generation.PopulateResultTable(inputFolder, outputDir, htmltable)
    # html_generation.FinalizeHTML(htmlHeader, r_pass, r_fail, htmltable, summaryFileName)
