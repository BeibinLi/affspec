import affspec
import cv2
import glob

# %% Create a session
# The backbone can be either "esp" or "mob" or "res", where the default is "esp"
session = affspec.pipeline.Process(backbone="esp")

# %% Process a numpy image
img = cv2.imread("images/beibin.jpg")
rst = session.run_one_img(img)
print(rst)


# You can access each value using the dictionary
print(rst["expression"])
print(rst["valence"])
print(rst["arousal"])
print(rst["action units"])


# a more human-readable action units format
au_description = affspec.config.au_array_2_description(rst["action units"])
print(au_description)

# %% Process a image by passing an image path
rst = session.run_one_img(imgname="images/beibin.jpg")
print(rst)

# %% If there is no face in the image. 
# The result will be None
rst = session.run_one_img(imgname="images/no_face.jpg")
print(rst)

# %% Process several images at one time
imgs = glob.glob("images/*.jpg")
print(imgs)

rsts = session.run_imgs(imgs)
print(rsts)

print("-" * 50, "\n")
[print(_) for _ in rsts]

