from helperFunctions import *
from faceMorpher import *
import aligner
from aligner import *

imageLocation = '/Users/me/Desktop/MITDentalData/Photos/'
for i in range (2,52):
	img_name = imageLocation + str(i) +'i.jpg'
	print 'image name',img_name
	img = cv2.imread(img_name)

	img,points = landmark_locator(img_name,  width=500, height=600, fps=10)

	#point 38, 39
	#img = img[point[38][1]+10,points[38][0]+30]
	new_img = img
	new_img[points[38,1]-80:points[38,1]+80, :] = 0
	#cv2.line(img,(points[38][0],points[38][1]),(points[38][0],points[38][1]),(0,0,255),3)
	#cv2.line(img,(points[39][0],points[39][1]),(points[39][0],points[39][1]),(0,0,255),3)
	cv2.imshow('img',new_img)
#	cv2.waitKey(0)
	
	directoryLocation = os.path.dirname(os.path.abspath(__file__))
	cv2.imwrite(imageLocation+'blocked'+str(i)+'i.jpeg',new_img)
	cv2.destroyAllWindows()