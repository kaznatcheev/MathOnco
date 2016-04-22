import numpy as np
import cv2

def FluorescentAreaMark(img):
    #1: contrast limited adaptive histogram equalization to get rid of glow
    clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(20,20))

    clahe_img = clahe.apply(img)
    
    #2: threshold and clean out salt-pepper noise
    ret, thresh_img = cv2.threshold(clahe_img,127,255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)

    clean_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    return clean_img

def AreaCount(col_let, row_num, fot_num, dirName = '', ret_img = True):
    #load the images
    head = dirName + col_let + str(row_num) + '-' + str(fot_num)
    
    rdImage = cv2.imread(head + '-C1.tif',cv2.IMREAD_UNCHANGED)
    gnImage = cv2.imread(head + '-C2.tif',cv2.IMREAD_UNCHANGED)
    if ret_img:
        grImage = cv2.imread(head + '-P.tif',cv2.IMREAD_UNCHANGED)
    
    #switch to 8bit with proper normalizing
    if np.amax(rdImage) > 255:
        rdImage8 = cv2.convertScaleAbs(rdImage, alpha = (255.0/np.amax(rdImage)))
    else:
        rdImage8 = cv2.convertScaleAbs(rdImage)
        
    if np.amax(gnImage) > 255:    
        gnImage8 = cv2.convertScaleAbs(gnImage, alpha = (255.0/np.amax(gnImage)))
    else:
        gnImage8 = cv2.convertScaleAbs(gnImage)
    
    #get the area masks
    rdFA = FluorescentAreaMark(rdImage8)
    gnFA = FluorescentAreaMark(gnImage8)
    
    ign_buf = 30 #how big of an edge do we ignore?
    rd_area = (rdFA[ign_buf:-ign_buf,ign_buf:-ign_buf] > 0)
    gn_area = (gnFA[ign_buf:-ign_buf,ign_buf:-ign_buf] > 0)
    
    #create image to save
    img_out = []
    if ret_img:
        bW = 0.85
        
        img_out = cv2.merge((
            cv2.addWeighted(grImage,bW,rdFA,1- bW,1),
            cv2.addWeighted(grImage,bW,gnFA,1 - bW,1),
            cv2.addWeighted(grImage,bW,np.zeros_like(grImage), 1 - bW, 1)))
    
    return [np.sum(rd_area),np.sum(gn_area)], img_out
