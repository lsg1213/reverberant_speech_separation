import os

def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print(dirs)
