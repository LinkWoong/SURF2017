import os
from PIL import Image

path = '/media/linkwong/D/Container'

for filename in os.listdir(path):

    img = Image.open(os.path.join(path, filename))
    img = img.resize((256, 256), Image.ANTIALIAS)
    img.save(os.path.join(path, filename))

