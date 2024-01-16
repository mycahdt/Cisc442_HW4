import cv2 as cv
import numpy as np
import os


# Function edgeDectoror()
# Runs an in-built edge detector and produces an output image with edges
def edgeDetector():
  
    # Reads the image
    image = cv.imread("images/flower_HW4.jpg")

    #operatedImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Use the Canny Edge filter
    edge = cv.Canny(image, 100, 200, 3)
    
    # Write to the file edgesFlower.jpg for the Output image, which has edges
    cv.imwrite("images/edgesFlower.jpg", edge)

    # Shows Original Image
    cv.imshow('Original Image', image)

    # Shows Image with edges
    cv.imshow('Image with Edges', edge)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function cornerDectoror()
# Runs an in-built corner detector and produces an output image with corners
def cornerDetector():

    # Reads the image
    image = cv.imread("images/flower_HW4.jpg")
    
    # Changes the input image into a grayscale color space
    operatedImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Changes the data type setting to a 32-bitt floating point
    operatedImage = np.float32(operatedImage)
    
    # Uses the cornerHarris filter to detect the corners
    dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Uses dilate() to mark the resulting corners
    dest = cv.dilate(dest, None)
    
    # Uses the original image to use the optimal threshold value
    image[dest > 0.01 * dest.max()]=[0, 255, 0]

    # Write to the file cornersFlower.jpg for the output image, which has corners
    cv.imwrite("images/cornersFlower.jpg", image)
    
    # Shows Image with Corners
    cv.imshow('Image with Corners', image)
    
    # De-allocate any associated memory usage 
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    
# Function rotateEdgesImg()
# Rotates the image by 60 degrees
# And runs an in-built edge detector and produces an output image with edges
def rotateEdgesImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses getRotationMatrix2D() and warpAffine() to rotate image by 60 degrees
    rows,cols,channels= image.shape 
    M = cv.getRotationMatrix2D((cols/2,rows/2),60,1) 
    rotate_60 = cv.warpAffine(image,M,(cols,rows)) 


    # Uses Canny filter to get image with edges
    edgeImg = cv.Canny(rotate_60, 100, 200, 3) 

    # Write to the file rotatedEdgesFlower.jpg for the output image, 
    # which is rotated 60 degrees and has edges
    cv.imwrite("images/rotatedEdgesFlower.jpg", edgeImg)
    
    # Shows Image Rotated by 60 degres and has Edges
    cv.imshow('Image Rotated by 60 Degrees with Edges', edgeImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function rotateCornersImg()
# Rotates the image by 60 degrees
# And runs an in-built corner detector and produces an output image with corners
def rotateCornersImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses getRotationMatrix2D() and warpAffine() to rotate image by 60 degrees
    rows,cols,channels= image.shape 
    M = cv.getRotationMatrix2D((cols/2,rows/2),60,1) 
    rotate_60 = cv.warpAffine(image,M,(cols,rows)) 

    
    # Changes the input image into a grayscale color space
    operatedImage = cv.cvtColor(rotate_60, cv.COLOR_BGR2GRAY)
    
    # Changes the data type setting to a 32-bitt floating point
    operatedImage = np.float32(operatedImage)
    
    # Uses the cornerHarris filter to detect the corners
    dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Uses dilate() to mark the resulting corners
    dest = cv.dilate(dest, None)
    
    # Uses the original image to use the optimal threshold value
    rotate_60[dest > 0.01 * dest.max()]=[0, 255, 0]

    # Write to the file rotatedEdgesFlower.jpg for the output image, 
    # which is rotated 60 degrees and has corners
    cv.imwrite("images/rotatedCornersFlower.jpg", rotate_60)
    
    # Shows Image Rotated by 60 degres and has Corners
    cv.imshow('Image Rotated by 60 Degrees with Corners', rotate_60)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function scaleEdgesImg()
# Scales the image by 1.6 in the x and y directions
# And runs an in-built edge detector and produces an output image with edges
def scaleEdgesImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses resize() to scale the image by 1.6 in the x and y directions
    resizedImg = cv.resize(image, None, fx = 1.6, fy = 1.6, interpolation=cv.INTER_LINEAR)

    # Uses Canny filter to get image with edges
    edgeImg = cv.Canny(resizedImg, 100, 200, 3) 

    # Write to the file scaledEdgesFlower.jpg for the output image, 
    # which is scaled by 1.6 in the x and y directions and has edges
    cv.imwrite("images/scaledEdgesFlower.jpg", edgeImg)
    
    # Shows Image Scaled by 1.6 and has Edges
    cv.imshow('Image Scaled by 1.6 with Edges', edgeImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function scaleCornersImg()
# Scales the image by 1.6 in the x and y directions
# And runs an in-built corner detector and produces an output image with corners
def scaleCornersImg():
    
    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses resize() to scale the image by 1.6 in the x and y directions
    resizedImg = cv.resize(image, None, fx = 1.6, fy = 1.6, interpolation=cv.INTER_LINEAR)

    
    # Changes the input image into a grayscale color space
    operatedImage = cv.cvtColor(resizedImg, cv.COLOR_BGR2GRAY)
    
    # Changes the data type setting to a 32-bitt floating point
    operatedImage = np.float32(operatedImage)
    
    # Uses the cornerHarris filter to detect the corners
    dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Uses dilate() to mark the resulting corners
    dest = cv.dilate(dest, None)
    
    # Uses the original image to use the optimal threshold value
    resizedImg[dest > 0.01 * dest.max()]=[0, 255, 0]

    # Write to the file scaledCornersFlower.jpg for the output image, 
    # which is scaled by 1.6 in the x and y directions and has corners
    cv.imwrite("images/scaledCornersFlower.jpg", resizedImg)
    
    # Shows Image Scaled by 1.6 and has Corners
    cv.imshow('Image Scaled by 1.6 with Corners', resizedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()



# Function shearXEdgesImg()
# Scales the image by 1.2 in the x direction
# And runs an in-built edge detector and produces an output image with edges
def shearXEdgesImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses warpPerspective() to shear the image in the x direction by 1.2
    rows, cols, dim = image.shape
    M = np.float32([[1, 1.2, 0], [0, 1, 0], [0, 0, 1]])
    shearedImg = cv.warpPerspective(image, M, (int(cols*2), int(rows*1.2)))

    # Uses Canny filter to get image with edges
    edgeImg = cv.Canny(shearedImg, 100, 200, 3) 

    # Write to the file shearedXEdgesFlower.jpg for the output image, 
    # which is scaled by 1.2 in the x direction and has edges
    cv.imwrite("images/shearedXEdgesFlower.jpg", edgeImg)
    
    # Shows Image Scaled by 1.2 and has Edges
    cv.imshow('Image Scaled by 1.2 in x Direction with Edges', edgeImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function shearXCornersImg()
# Scales the image by 1.2 in the x direction
# And runs an in-built corner detector and produces an output image with corners
def shearXCornersImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses warpPerspective() to shear the image in the x direction by 1.2
    rows, cols, dim = image.shape
    M = np.float32([[1, 1.2, 0], [0, 1, 0], [0, 0, 1]])
    shearedImg = cv.warpPerspective(image, M, (int(cols*2), int(rows*1.2)))

    # Changes the input image into a grayscale color space
    operatedImage = cv.cvtColor(shearedImg, cv.COLOR_BGR2GRAY)
    
    # Changes the data type setting to a 32-bitt floating point
    operatedImage = np.float32(operatedImage)
    
    # Uses the cornerHarris filter to detect the corners
    dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Uses dilate() to mark the resulting corners
    dest = cv.dilate(dest, None)
    
    # Uses the original image to use the optimal threshold value
    shearedImg[dest > 0.01 * dest.max()]=[0, 255, 0]

    # Write to the file shearedXCornersFlower.jpg for the output image, 
    # which is scaled by 1.2 in the x direction and has corners
    cv.imwrite("images/shearedXCornersFlower.jpg", shearedImg)
    
    # Shows Image Scaled by 1.2 and has Corners
    cv.imshow('Image Scaled by 1.2 in x Direction with Corners', shearedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()






# Function shearYEdgesImg()
# Scales the image by 1.4 in the y direction
# And runs an in-built edge detector and produces an output image with edges
def shearYEdgesImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses warpPerspective() to shear the image in the y direction by 1.4
    rows, cols, dim = image.shape
    M = np.float32([[1, 0, 0],[1.4, 1, 0],[0, 0, 1]])
    shearedImg = cv.warpPerspective(image, M, (int(cols*1.2), int(rows*3)))

    # Uses Canny filter to get image with edges
    edgeImg = cv.Canny(shearedImg, 100, 200, 3) 

    # Write to the file shearedXEdgesFlower.jpg for the output image, 
    # which is scaled by 1.4 in the y direction and has edges
    cv.imwrite("images/shearedYEdgesFlower.jpg", edgeImg)
    
    # Shows Image Scaled by 1.4 and has Edges
    cv.imshow('Image Scaled by 1.4 in y Direction with Edges', edgeImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Function shearYCornersImg()
# Scales the image by 1.4 in the y direction
# And runs an in-built corner detector and produces an output image with corners
def shearYCornersImg():

    # Reads the image
    image = cv.imread('images/flower_HW4.jpg')

    # Uses warpPerspective() to shear the image in the y direction by 1.4
    rows, cols, dim = image.shape
    M = np.float32([[1, 0, 0],[1.4, 1, 0],[0, 0, 1]])
    shearedImg = cv.warpPerspective(image, M, (int(cols*1.2), int(rows*3)))

    # Changes the input image into a grayscale color space
    operatedImage = cv.cvtColor(shearedImg, cv.COLOR_BGR2GRAY)
    
    # Changes the data type setting to a 32-bitt floating point
    operatedImage = np.float32(operatedImage)
    
    # Uses the cornerHarris filter to detect the corners
    dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Uses dilate() to mark the resulting corners
    dest = cv.dilate(dest, None)
    
    # Uses the original image to use the optimal threshold value
    shearedImg[dest > 0.01 * dest.max()]=[0, 255, 0]

    # Write to the file shearedYCornersFlower.jpg for the output image, 
    # which is scaled by 1.4 in the y direction and has corners
    cv.imwrite("images/shearedYCornersFlower.jpg", shearedImg)
    
    # Shows Image Scaled by 1.4 and has Corners
    cv.imshow('Image Scaled by 1.4 in y Direction with Corners', shearedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():

    # Note: Please check the image's actual dimensions to see the image being scaled

    # ------------------------------------------------------------------------------------------------------------------------
    # Problem 1: Run an in-built edge detector and a corner detector, 
    # and produce two output images: (i) image with edges, ii) image with corners.
    edgeDetector()         # See edgesFlower.jpg for output image
    cornerDetector()       # See cornersFlower.jpg for output image

    # ------------------------------------------------------------------------------------------------------------------------
    # Problem 2: Rotate the original image by 60-degrees and perform (1)
    rotateEdgesImg()       # See rotatedEdgesFlower.jpg for output image
    rotateCornersImg()     # See rotatedCornersFlower.jpg for output image

    # ------------------------------------------------------------------------------------------------------------------------
    # Problem 3: Scale the original image by 1.6 in both the x and y-directions and perform (1)
    scaleEdgesImg()        # See scaledEdgesFlower.jpg for output image       
    scaleCornersImg()      # See scaledCornersFlower.jpg for output image

    # ------------------------------------------------------------------------------------------------------------------------
    # Problem 4: Shear the original image in the x-direction by 1.2 and perform (1)
    shearXEdgesImg()        # See shearedXEdgesFlower.jpg for output image       
    shearXCornersImg()      # See shearedXCornersFlower.jpg for output image

    # ------------------------------------------------------------------------------------------------------------------------
    # Problem 5: Shear the original image in the y-direction by 1.4 and perform (1)
    shearYEdgesImg()        # See shearedYEdgesFlower.jpg for output image       
    shearYCornersImg()      # See shearedYCornersFlower.jpg for output image

if __name__ == '__main__':
    main()