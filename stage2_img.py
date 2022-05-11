
import cv2
import os
import subprocess
import glob
import os
from tqdm import tqdm
import numpy as np 

img_path = glob.glob('bundler-v0.3-binary/Photos/*.jpg')
from PIL import Image


for path in tqdm(img_path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img_name = os.path.basename(path)
    try:
        _ = img.shape
        _ = Image.fromarray(np.uint8(img)).convert('RGB')
        check = _.verify()

        if check is None:
            subprocess.call([
                f'cp {path} ' + os.path.join('Photo_new', img_name)], shell=True)

    except:
        print('PASS')
        pass






