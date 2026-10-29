"""Educational sky-image heuristic, NOT a validated cloud-cover model. Author: Harpreet Singh."""
import io
import numpy as np
from PIL import Image
def sky_image_analysis(payload:bytes):
    if len(payload)>8*1024*1024:raise ValueError('Maximum image size is 8 MB')
    with Image.open(io.BytesIO(payload)) as im:
        im.verify()
    with Image.open(io.BytesIO(payload)) as im:
        im=im.convert('RGB');im.thumbnail((1024,1024));a=np.asarray(im,dtype=np.float32)/255
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    brightness=a.mean(axis=2)
    blue_sky=(b>r*1.15)&(b>g*1.05)&(b>0.3)
    gray_bright=(np.max(a,axis=2)-np.min(a,axis=2)<0.17)&(brightness>0.55)
    return {'blue_like_pct':float(blue_sky.mean()*100),'bright_gray_like_pct':float(gray_bright.mean()*100),'mean_brightness':float(brightness.mean()),'preview':Image.fromarray((a*255).astype('uint8'))}
