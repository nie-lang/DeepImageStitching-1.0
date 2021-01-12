from tempfile import TemporaryFile
from PIL import Image
import numpy as np
import glob
import os.path
import random as rd
import cv2
import matplotlib.pyplot as plt
import math


#Calculate cross product
def cross(a, b, c):
    ans = (b[0] - a[0])*(c[1] - b[1]) - (b[1] - a[1])*(c[0] - b[0])
    return ans

#Check whether the quadrilateral is convex.
#If it is convex, return 1
def checkShape(a,b,c,d):
    x1 = cross(a,b,c)
    x2 = cross(b,c,d)
    x3 = cross(c,d,a)
    x4 = cross(d,a,b)

    if (x1<0 and x2<0 and x3<0 and x4<0) or (x1>0 and x2>0 and x3>0 and x4>0) :
        return 1
    else:
        print('not convex')
        return 0


#Judge whether the pixel is within the label, and set it to black if it is not
def inLabel(row, col, src_input, dst):
    if (row >= src_input[0][1]) and (row <= src_input[2][1]) and (col >= src_input[0][0]) and (col <= src_input[1][0]) :
        return 1
    else : 
        #Only handle the case of convex quadrilaterals. As for the concave quadrilateral, regenerated it until it is convex.
        a = (dst[1][0] - dst[0][0])*(row - dst[0][1]) - (dst[1][1] - dst[0][1])*(col - dst[0][0])
        b = (dst[3][0] - dst[1][0])*(row - dst[1][1]) - (dst[3][1] - dst[1][1])*(col - dst[1][0])
        c = (dst[2][0] - dst[3][0])*(row - dst[3][1]) - (dst[2][1] - dst[3][1])*(col - dst[3][0])
        d = (dst[0][0] - dst[2][0])*(row - dst[2][1]) - (dst[0][1] - dst[2][1])*(col - dst[2][0])
        if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0) :
            return 1
        else :
            return 0



# Load a random image from the dataset
def load_random_image(path_source, size):
    #The size of the randomly sampled image must be greater than width*height
    img_path = rd.choice(glob.glob(os.path.join(path_source, '*.jpg'))) 
    img = Image.open(img_path)
    while True:
        #print(img.size)
        if img.size[0]>=size[0] and img.size[1]>=size[1] :
            break
        img_path = rd.choice(glob.glob(os.path.join(path_source, '*.jpg'))) 
        img = Image.open(img_path)
    #print('bingo')
    img_grey = img.resize(size)                
    img_data = np.asarray(img_grey)
    #imggg = Image.fromarray(img_data.astype('uint8')).convert('RGB')
    #imggg.show()
    return img_data


def save_to_file(index, image1, image2, label, path_dest, gt_4pt_shift):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    input1_path = path_dest +'//input1' 
    input2_path = path_dest +'//input2'
    label_path = path_dest +'//label'
    pt4_shift_path = path_dest + '//shift'
    
    if not os.path.exists(input1_path):
        os.makedirs(input1_path)
    if not os.path.exists(input2_path):
        os.makedirs(input2_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    if not os.path.exists(pt4_shift_path):
        os.makedirs(pt4_shift_path)

    input1_path = path_dest +'//input1//' + index + '.jpg'
    input2_path = path_dest +'//input2//'+ index + '.jpg'
    label_path = path_dest +'//label//'+ index + '.jpg'
    pt4_shift_path = path_dest +'//shift//'+ index + '.npy'
    image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
    image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
    label = Image.fromarray(label.astype('uint8')).convert('RGB')
    image1.save(input1_path)
    image2.save(input2_path)
    label.save(label_path)
    np.save(pt4_shift_path, gt_4pt_shift)


    

# Function to generate dataset
def generate_dataset(path_source, path_dest, rho, height, width, data, box, overlap):
    
    for count in range(0, data):
        #load row image
        img = load_random_image(path_source, [width, height]).astype(np.uint16)

        #define parameters
        #src_input1 = np.empty([4, 2], dtype=np.uint8)
        src_input1 = np.zeros([4, 2])
        src_input2 = np.zeros([4, 2])
        dst = np.zeros([4, 2])

        #Upper left
        src_input1[0][0] = int(width/2 - box/2)
        src_input1[0][1] = int(height/2 - box/2)
        # Upper right
        src_input1[1][0] = src_input1[0][0] + box
        src_input1[1][1] = src_input1[0][1]
        # Lower left
        src_input1[2][0] = src_input1[0][0]
        src_input1[2][1] = src_input1[0][1] + box
        # Lower right
        src_input1[3][0] = src_input1[1][0]
        src_input1[3][1] = src_input1[2][1]
        #print(src_input1)

        #The translation of input2 relative to input1
        box_x_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
        box_y_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
        #Upper left
        src_input2[0][0] = src_input1[0][0] + box_x_off
        src_input2[0][1] = src_input1[0][1] + box_y_off
        #Upper righ
        src_input2[1][0] = src_input1[1][0] + box_x_off
        src_input2[1][1] = src_input1[1][1] + box_y_off
        # Lower left
        src_input2[2][0] = src_input1[2][0] + box_x_off
        src_input2[2][1] = src_input1[2][1] + box_y_off
        #Lower right
        src_input2[3][0] = src_input1[3][0] + box_x_off
        src_input2[3][1] = src_input1[3][1] + box_y_off
        #print(src_input2)

        offset = np.empty(8, dtype=np.int8)
        # Generate offsets:
        #The position of each vertex after the coordinate perturbation
        while True:
            for j in range(8):
                offset[j] = rd.randint(-rho, rho)
            # Upper left
            dst[0][0] = src_input2[0][0] + offset[0]
            dst[0][1] = src_input2[0][1] + offset[1]
            # Upper righ
            dst[1][0] = src_input2[1][0] + offset[2]
            dst[1][1] = src_input2[1][1] + offset[3]
            # Lower left
            dst[2][0] = src_input2[2][0] + offset[4]
            dst[2][1] = src_input2[2][1] + offset[5]
            # Lower right
            dst[3][0] = src_input2[3][0] + offset[6]
            dst[3][1] = src_input2[3][1] + offset[7]
            #print(dst)
            if checkShape(dst[0],dst[1],dst[3],dst[2])==1 :
                break

        source = np.zeros([4, 2])
        target = np.zeros([4, 2])
        source[0][0] = 0
        source[0][1] = 0
        source[1][0] = source[0][0] + box
        source[1][1] = source[0][1]
        source[2][0] = source[0][0]
        source[2][1] = source[0][1] + box
        source[3][0] = source[1][0]
        source[3][1] = source[2][1]
        target[0][0] = dst[0][0] - src_input1[0][0]
        target[0][1] = dst[0][1] - src_input1[0][1]
        target[1][0] = dst[1][0] - src_input1[0][0]
        target[1][1] = dst[1][1] - src_input1[0][1]
        target[2][0] = dst[2][0] - src_input1[0][0]
        target[2][1] = dst[2][1] - src_input1[0][1]
        target[3][0] = dst[3][0] - src_input1[0][0]
        target[3][1] = dst[3][1] - src_input1[0][1]
        Hab, status = cv2.findHomography(source, target)

        # Generate the shift
        gt_4pt_shift = np.zeros((8,1), dtype = np.float32)
        for i in range(4):
        	gt_4pt_shift[2*i] = target[i][0] - source[i][0]
        	gt_4pt_shift[2*i+1] = target[i][1] - source[i][1]


        h, status = cv2.findHomography(dst, src_input2)
        img_warped = np.asarray(cv2.warpPerspective(img, h, (width, height))).astype(np.uint8)

        # Generate the label
        label = img.copy()
        for row in range(0, img.shape[0]):
            for col in range(0, img.shape[1]):
                if not (inLabel(row, col, src_input1, dst)):
                    label[row][col] = 0

        x = int(src_input1[0][0] - overlap * box - rho)
        y = int(src_input1[0][1] - overlap * box - rho)
        label = label[y+1:int(y + box + 2 * overlap * box + 2 * rho-1), x+1:int(x + box + 2 * overlap * box + 2 * rho-1)]


        
        # Generate input1
        x1 = int(src_input1[0][0])
        y1 = int(src_input1[0][1])
        image1 = img[y1:y1+box, x1:x1+box]
        
        # Generate input2
        x2 = int(src_input2[0][0])
        y2 = int(src_input2[0][1])
        image2 = img_warped[y2:y2+box, x2:x2+box,...]
        
        save_to_file(str(count+1).zfill(6), image1, image2, label, path_dest, gt_4pt_shift)
        print(count+1)
        
        



raw_image_path = 'D://dataset//COCO//val2014'       
box_size = 128
height = 360
width = 480
overlap_rate = 0.5 
rho = int(box_size/5.0)

### generate training dataset
print("Training dataset...")
dataset_size = 20
generate_image_path = 'D://My Projects//VFIS-Net//dataset//training'
generate_dataset(raw_image_path, generate_image_path, rho, height, width, dataset_size, box_size, overlap_rate)
### generate testing dataset
print("Testing dataset...")
dataset_size = 10
generate_image_path = 'D://My Projects//VFIS-Net//dataset//testing'
generate_dataset(raw_image_path, generate_image_path, rho, height, width, dataset_size, box_size, overlap_rate)


