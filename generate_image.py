from faker import Faker
import re
from torch_snippets.loader import * # !pip install torch_snippets
from cv2utils import *
from datetime import datetime
from PIL import Image
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
from itertools import dropwhile
import random

ldrop = lambda fn, x: list(dropwhile(fn, x))
lrev = lambda x: list(reversed(x))
gBlur = lambda x, k=5: cv2.GaussianBlur(x, (k,k), k)
chop = lambda x, fn=lambda x: x==0: np.array(lrev(ldrop(fn, lrev(ldrop(fn, x)))))
clean_img = lambda img: 1 - chop(chop(1-img, fn=lambda x: x.sum()==0).T, fn=lambda x: x.sum()==0).T
clean_img = lambda img: 1 - chop(chop(1-img, fn=lambda x: x.sum()==0).T, fn=lambda x: x.sum()==0).T

def make_image(text, font=None, font_sz=100):
    '''Primitive version of `create_image_char_by_char`
    Fast, but no scope for augmentation and doesn't return bbs
    '''
    img = Image.new("RGBA", (1,1))
    draw = ImageDraw.Draw(img)
    if isinstance(font, str): font = ImageFont.truetype(font, font_sz)
    w, h = draw.textsize(text, font)
    sz = (w+100,h+100)
    if isinstance(font, str): font = ImageFont.truetype(font, 100)
    img = Image.new('RGB', sz, color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, font=font, fill=(0,0,0))
    img = np.array(img.convert(mode='1')).astype(np.uint8)
    return clean_img(img)


from PIL import Image, ImageDraw, ImageFont
def create_image_char_by_char(text, font, aug=None, font_sz=100):
    '''Create image and return bbs for every character
    Optionally, every character can be augmented to achieve added randomness
    >>> im, bbs = create_image_char_by_char('sdlfljk\nsldfkj sdkfj', font=font)
    >>> show(im, bbs=bbs)
    '''
    img = Image.new("RGBA", (1,1))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, font_sz)
    w, h = draw.textsize(text, font)
    textsize = (w+100,h+100)
    img = Image.new('RGB', textsize, color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, font=font, fill=(0,0,0))
    bbs = []
    for line_no, _text in enumerate(text.split('\n')):
        for i, char in enumerate(_text):
            _, Y_1 = font.getsize(_text[i])
            X, Y_2 = font.getsize(_text[:i+1])
            Y = Y_1 if Y_1 < Y_2 else Y_2
            # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageDraw.py#L374
            Y = Y + line_no*(draw.textsize("A", font=font)[1] + 4)
            X += 10; Y += 10
            width, height = font.getmask(char).size
            y = Y - height
            x = X - width
            bbs.append(BB(x,y,X,Y))
    _img = 255*np.array(img.convert(mode='1')).astype(np.uint8)
    if aug is not None:
        IMG = 255*np.ones_like(_img).astype(np.uint8)
        for char, bb in zip(text.replace('\n',''), bbs):
            # try:
                x,y,X,Y = bb
                if (bb.h == 0 or bb.w == 0): continue
                if char in [' ','\n']: continue
                cx, cy = (bb.x+bb.X)/2, (bb.y+bb.Y)/2
                crop = uint(make_image(char, font=font, font_sz=font_sz))
                crop = aug(images=[crop])[0]
                h, w = crop.shape
                _bb = BB(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
                nzpxls = ys, xs = np.nonzero(255 - crop)
                nzpxls_IM = ys+y, xs+x
                IMG[nzpxls_IM] = crop[nzpxls]
            # except Exception as e:
            #     logger.warning(e)
        _img = IMG.copy()
    _img, (_x,_y,_X,_Y) = get_bounding_image(_img)
    bbs = [BB(x-_x,y-_y,X-_x,Y-_y) for x,y,X,Y in bbs]
    return _img, bbs

def patch(im, IM, origin=None):
    '''patch `im` into `IM` (inplace) at `origin (x,y)`
    patch ONLY the black pixels
    '''
    try:
        h, w = im.shape
        H, W = IM.shape
        if origin is not None:
            x, y = origin 
        else:
            x, y = randint(W-w), randint(H-h)
        nzpxls = ys, xs = np.nonzero(255 - im)
        nzpxls_IM = ys+y, xs+x
        IM[nzpxls_IM] = im[nzpxls]
    except Exception as e:
        print(e)
        return -1
