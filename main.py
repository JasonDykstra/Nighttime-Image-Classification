import cv2
import numpy as np
import glob
import random
from typing import Optional

DISTANCE_THRESHOLD = 0.75
MATCH_THRESHOLD = 30

def subtract(img1, img2):
    """Helper function that subtracts two images of different sizes. Channels must be the same however, I suggest importing as greyscale."""

    # Get sizes of images
    dim1 = img1.shape
    dim2 = img2.shape
    size1 = dim1[0] * dim1[1]
    size2 = dim2[0] * dim2[1]

    # If sizes are same, return subtracted image
    if size1 == size2:
        return cv2.subtract(img1, img2)

    # Scale down larger image to size of smaller image
    # (Subtraction requires images to be exact same size)
    if size1 > size2:
        img1 = cv2.resize(img1, (dim2[1], dim2[0]))
    else:
        img2 = cv2.resize(img2, (dim1[1], dim1[0]))


    # Return subtraction of both images
    return cv2.subtract(img1, img2)

def compare(img1, img2, same: Optional[bool] = None):
    """Helper function for comparing images. Using K-nearest neighbors, this function compares key points and prints whether or not the images are a likely match."""

    # Create ORB feature detector object, nfeatures 500 by default
    orb = cv2.ORB_create(nfeatures=500)

    # Create key points and descriptor objects for the images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create brute force matcher object
    bf = cv2.BFMatcher()
    # Match the descriptors between day and night images
    matches = bf.knnMatch(des1, des2, k=2)

    # Find "good" matches, i.e. find pairs of descripts that are relatively close together, k-nearest neighbors method
    good = []
    for m, n in matches:
        if m.distance < DISTANCE_THRESHOLD * n.distance:
            good.append([m])

    # Draw the matches from the knn result
    matchKNN = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    # Store the result of the comparison
    match = True if len(good) >= MATCH_THRESHOLD else False
    
    # Create window title
    title = "Images are a match!" if match else "Images are different!"
    if same is not None:
        title = "Correct!" if match == same else "Incorrect."
    
    # Draw the comparison of keypoints across the first and second images, with the match status of the images as the window title
    cv2.imshow(title, matchKNN)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testImages():
    """Function that runs through a bunch of test images and compares them, mixed day and night."""

    dayImages = []
    nightImages = []

    dayImages.extend([[cv2.imread(file), 1] for file in glob.glob("images/test/00021510/day/*.jpg")])
    dayImages.extend([[cv2.imread(file), 2] for file in glob.glob("images/test/00023966/day/*.jpg")])
    
    nightImages.extend([[cv2.imread(file), 1] for file in glob.glob("images/test/00021510/night/*.jpg")])
    nightImages.extend([[cv2.imread(file), 2] for file in glob.glob("images/test/00023966/night/*.jpg")])

    # List structure: [[img1, 1], [img2, 1], [img3, 1]]
    # Where the second element is which set it came from

    # Helper function to pop a random element from the given list
    def popRandom(list):
        i = random.randrange(0, len(list))
        return list.pop(i)

    # Create pairs of day and night images until one or both lists has zero elements left
    pairs = []
    while dayImages and nightImages:
        randDay = popRandom(dayImages)
        randNight = popRandom(nightImages)
        pair = [randDay[0], randNight[0], randDay[1] == randNight[1]]
        pairs.append(pair)

    # Make the second image of each pair the subtraction of the two
    for pair in pairs:
        pair[1] = subtract(pair[0], pair[1])

    # Compare the images in the pairs against each other (Currently only limited to 10 images)
    counter = 0
    for pair in pairs:
        if counter == 10:
            break
        compare(pair[0], pair[1], pair[2])
        counter += 1


testImages()
