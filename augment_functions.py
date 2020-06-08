'v1.0'
import imgaug.augmenters as iaa
import imgaug as ia
sometimes = lambda aug, p=0.5: iaa.Sometimes(p, aug)
from functools import partial
from cv2utils import *

def get_bounding_image(im):
    bbs = get_bbs(im)
    h,w = im.shape[:2]
    if len(bbs) < 1: return im, (0,0,w,h)
    xs,ys,Xs,Ys = zip(*bbs)
    x,y,X,Y = min(xs),min(ys),max(Xs),max(Ys)
    x = max(0, x-1)
    return crop_from_bb(im, (x,y,X,Y)), (x,y,X,Y)
def tight_crop(images, random_state, parents, hooks):
    for ix, img in enumerate(images):
        images[ix] = get_bounding_image(img)
    return images
def identity_fn(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

def random_gray(images, random_state, parents, hooks):
    for ix, im in enumerate(images):
        h, w = im.shape[:2]
        rand_gray = gr = 180 + randint(75)
        rand_black = bl = randint(30)
        im[im > 180] = gr
        im[im < 120] = bl
        im[0,0] = 0
        im[h-1,w-1] = 255
        images[ix] = im
    return images

def random_underline(images, random_state, parents, hooks):
    for ix, im in enumerate(images):
        h, w = im.shape[:2]
        im = np.array(im)
        x1, x2 = 0, w
        y1, y2 = [h - randint(max(10, h//20)) for _ in range(2)]
        thickness = 2 + randint(10) if h > 30 else 2 + randint(4)
        fill = randint(40)
        cv2.line(im, (x1, y1), (x2, y2), (fill,)*3, thickness)
        images[ix] = im
    return images
        
def random_upperline(images, random_state, parents, hooks):
    for ix, im in enumerate(images):
        h, w = im.shape[:2]
        x1, x2 = 0, w
        y1, y2 = [randint(h//20) for _ in range(2)]
        thickness = 2 + randint(10) if h > 30 else 2 + randint(4)
        fill = randint(40)
        cv2.line(im, (x1, y1), (x2, y2), (fill,)*3, thickness)
        images[ix] = im
    return images

def random_border(images, random_state, parents, hooks):
    for ix, im in enumerate(images):
        h, w = im.shape[:2]
        x, X = randint(5), w - randint(5)
        y, Y = randint(5), h - randint(5)
        fill = randint(40)
        for start, end in [[(x, y), (x, Y)], [(x, Y), (X, Y)], [(X, Y), (X, y)], [(X, y), (x, y)]]:
            if randint(100) < 25: continue
            thickness = 1+randint(2) # 2 + randint(2) if h > 30 else 2 + randint(4)
            cv2.line(im, start, end, (fill,)*3, thickness)
        images[ix] = im
    return images

def pixellate(images, random_state, parents, hooks):
    for ix, im in enumerate(images):
        h, w = im.shape[:2]
        f = randint(4)
        _h, _w = int(h*f), int(w*f)
        im = cv2.resize(im, (_w, _h))
        im = cv2.resize(im, (w, h))
        images[ix] = im
    return images

TightCrop = iaa.Lambda(tight_crop, identity_fn)
RandomGray = iaa.Lambda(random_gray, identity_fn)
RandomUnderline = iaa.Lambda(random_underline, identity_fn)
RandomUpperline = iaa.Lambda(random_upperline, identity_fn)
RandomBorder = iaa.Lambda(random_border, identity_fn)
Pixellate = iaa.Lambda(pixellate, identity_fn)
