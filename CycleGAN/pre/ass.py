import os

def delete(path):

    for root, dirs, files in os.walk(path):

        for name in files:

            if name.endswith(".png"):

                os.remove(os.path.join(root, name))
                print "Delete File:{}".format(os.path.join(root,name))

path = '/media/linkwong/D/Container'

delete(path)

