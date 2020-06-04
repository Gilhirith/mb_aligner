# Some functions to visualize intermediate results if needed

import cv2
import numpy as np

def draw_keypoints_bounds(img, kps):
    img = img.copy()
    for kp in kps:
        center = np.array(kp.pt)
        p1 = np.round(center - kp.size).astype(np.int)
        p2 = np.round(center + kp.size).astype(np.int)

        cv2.rectangle(img, tuple(p1), tuple(p2), 255, thickness=3)

    return img

def draw_matches_downsampled(img1, kp1, img2, kp2, matches, ds=0.2):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    img1 = cv2.resize(img1, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    buffer = 20

    out = np.zeros((max([rows1,rows2]),cols1+cols2+buffer,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1+buffer:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = np.array(kp1[img1_idx].pt) * ds
        (x2,y2) = np.array(kp2[img2_idx].pt) * ds

        print("({},{}) -> ({},{})".format(x1, y1, x2, y2))
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        rand_color = np.random.rand(3,) * 255
        cv2.circle(out, (int(x1),int(y1)), 4, rand_color, 1)   
        cv2.circle(out, (int(x2)+cols1+buffer,int(y2)), 4, rand_color, 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1+buffer,int(y2)), rand_color, 3)


    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

    
