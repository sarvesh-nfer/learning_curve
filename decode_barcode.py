import zxing
import cv2
import numpy 
import pytesseract
from pylibdmtx.pylibdmtx import decode





# ==============================================================================
# BarcodeDecoding
# ==============================================================================


class BarcodeDecoding():
    '''Class to decode the barcode given path and image as input'''
# |----------------------------------------------------------------------------|
# class Variables
# |----------------------------------------------------------------------------|
    #no classVariables
# |----------------------------------------------------------------------------|
# Constructor
# |----------------------------------------------------------------------------|
    def __init__(self):
        self.outputString = ""
        self._imagepath = None
        self._contrast = 2
# |-------------------------End of Constructor------------------------------|
# ------------------------------------------------------------------------------#
# _preprocessingPipeline
# ------------------------------------------------------------------------------#
    def _preprocesPipeline(self, labelImage, contrast):
        
        """
        Input is an Label Image of the glass slide.
        The contrast varies with Exposure at which the Label Image is captured.
        The contrast parameter has to be moved to the Config XML file.
        """
        self._contrast = contrast

        contrast_img = cv2.addWeighted(labelImage, self._contrast, numpy.zeros(labelImage.shape, labelImage.dtype), 0, 0) 
        median = cv2.medianBlur(contrast_img,3)
        grayImage = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        return grayImage
# ------------------------------End of _preprocessingPipeline-----------------------#

# |----------------------------------------------------------------------------|
# processPipeline
# |----------------------------------------------------------------------------|
    def processPipeline(self, path):
        is_yellow_label = 1
        retry_count = 2
        trying_count = 0

        barcode_serial_number = None
        barcode_format_name = ""



        try:
            #Initialize the input variables
            self._imagepath = path
            img = cv2.imread(self._imagepath)

            # if is_yellow_label is True or (is_yellow_label is False and trying_count > 0):

                # | xzing barcode decoders requires image path as an input |
            reader = zxing.BarCodeReader()
            barcode = reader.decode(self._imagepath, True) 
            print('barcode ', barcode)

            if barcode != None:
                barcode_data = barcode.raw
                barcode_format = barcode.format
                print('Format of the barcode ', type(barcode_format))
                barcode = barcode_data.split(';')[0]
                self.outputString=''
                i=0
                for lts in barcode:
                    if 48 <= ord(lts) <= 57 or 65 <= ord(lts) <= 90 or 97 <= ord(lts) <= 122:
                        self.outputString +=barcode_data[i]
                    else:
                        self.outputString += '_'
                    i=i+1
                    
                print('zxing output : ', self.outputString)
                barcode_serial_number = self.outputString
                barcode_format_name = barcode_format

                # return self.outputString, barcode_format

            else:
                data_matrix_data = decode(img)
                print('Raw output ', data_matrix_data[0])
                data_matrix = (data_matrix_data[0].data)
                data_matrix_output = str(data_matrix)
                data_matrix_data = data_matrix_output.split("'")[1]
                
                print(type(data_matrix_output))
                
                print('data matrix ', data_matrix_data)
                
                data_matrix_output_final = ""
                for special_char in data_matrix_data:
                    if special_char.isalpha() is False and \
                        special_char.isdigit() is False:
                        special_char = "_"
                    data_matrix_output_final = data_matrix_output_final + special_char

                print('Final Decded Data Matrix ', data_matrix_output_final)
                barcode_type = 'Data_Matrix'

                barcode_serial_number = data_matrix_output_final
                barcode_format_name = barcode_type
                
                # return data_matrix_output_final, barcode_type
                if barcode_serial_number != None and barcode_serial_number != "":
                    return barcode_serial_number, barcode_format_name

            

        except Exception as msg:
            print("Exception occured in Barcode Decoding class, ", msg)
            

        return barcode_serial_number, barcode_format_name

# |----------------------End of processPipeline---------------------------|

# |----------------------------------------------------------------------------|
# main
# |----------------------------------------------------------------------------|
def main():
    # barcodePath = '/home/adminspin/Downloads/Mayo/505/BadSlides/2001V501005_11799/loc_output_data/barcodeImage.jpeg'
    barcodePath = input("Enter the path of the image : ")
    mBarcode = BarcodeDecoding()
    outputString = mBarcode.processPipeline(barcodePath)
    print("Output string: ", outputString)
    
# |----------------------End of main---------------------------|

if __name__ == '__main__':
    main()
