import os

path = '/media/linkwong/D/Container'

i = 0

for filename in os.listdir(path):

    newname = 'Train_' +str(i) + '.jpg'
    print newname

    i += 1
    os.rename(os.path.join(path,filename),os.path.join(path, newname))


