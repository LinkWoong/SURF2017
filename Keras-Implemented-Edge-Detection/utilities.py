def normal_map(image):
	image = image.astype(np.float)
	image = image / 255.0
	image = -image + 1 
	image[image < 0] = 0
	image[image > 1] = 1

	cv2.imshow('dick',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return image

def gray_map(image):

	gray = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2GRAY)
	highpass = gray.astype(np.float)
	highpass = highpass / 255.0
	highpass = -highpass + 1
	highpass = highpass[None]


	return highpass.transpose((1,2,0))
	#cv2.imshow('dick',highpass)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def light_map(image):

	gray = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(0,0),3)
	highpass = gray.astype(int) - blur.astype(int)
	highpass = highpass.astype(np.float)
	highpass = highpass / 128.0
	highpass = highpass[None]

	cv2.imshow('dick',blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return highpass.transpose((1,2,0))

def light_map_single(image):

	image = image[None]

	image = image.transpose((1,2,0))
	blur = cv2.GaussianBlur(image, (0,0), 2)
	image = image.reshape((image.shape[0],image.shape[1]))
	highpass = image.astype(int) - blur.astype(int)
	highpass = highpass.astype(np.float)
	highpass = highpass / 128.0

	cv2.imshow('dick',blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return highpass

def normalize_pic(image):

	image = image.astype(np.float)
	image = image / 255.0
	image = image / np.max(image)

	#cv2.imshow('dick',image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return image

def superlize_pic(image):

	image = normalize_pic(image)

	image = image * 2.333333
	image[image > 1] = 1

	cv2.imshow('dick',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	print image 
	return image
def resize_img(image):

	zeros = np.zeros((512,512,image.shape[2]), dtype=np.float)
	image = zeros[:image.shape[0],:image.shape[1]]

	return image
def show_active_img_and_save(name,image,path):

	image = image.astype(np.float)
	image = image / 255.0
	image = -image + 1
	mat = image  * 255.0

	mat[mat<0] = 0 
	mat[mat>255] = 255

	mat = mat.astype(np.uint8)

	cv2.imshow(name,mat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(path,mat)

	return image
def show_active_img_and_save_denoise(name,image,path):

	image = image.astype(np.float)
	image = image / 255.0
	image = -image + 1
	mat = image  * 255.0

	mat[mat<0] = 0 
	mat[mat>255] = 255

	mat = mat.astype(np.uint8)
	mat = ndimage.median_filter(mat,1)

	cv2.imshow(name,mat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(path,mat)

	return image
def show_active_img_and_save_denoise_filter(name,image,path):
	mat = image.astype(np.float)
	mat[mat<0.18] = 0
	mat = -mat + 1
	mat = mat * 255.0
	mat[mat<0] = 0
	mat[mat>255] = 255
	mat = mat.astype(np.uint8)

	mat = ndimage.median_filter(mat,1)
	cv2.imshow(name,mat)
	cv2.imwrite(path,mat)

	return mat


def show_active_img_and_save_denoise_filter2(name,image,path):
	mat = image.astype(np.float)
	mat[mat<0.1] = 0
	mat = -mat + 1
	mat = mat * 255.0
	mat[mat<0] = 0
	mat[mat>255] = 255
	mat = mat.astype(np.uint8)

	mat = ndimage.median_filter(mat,1)
	cv2.imshow(name,mat)
	cv2.imwrite(path,mat)

	return mat

def resize_img_512_3d(img):
    zeros = np.zeros((1,3,512,512), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0))
   