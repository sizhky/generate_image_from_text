'v1.01 (resize)'

'TODO: Add __all__'
import cv2, numpy as np
try: from .loader import *
except: from loader import *
from fuzzywuzzy import fuzz

def hlines(img, kernel_length=20, dilation_kernel=np.ones((1,1))):
    'Find horizontal lines in a kernel'
    kernel_length = int(np.array(img).shape[0]//kernel_length)

    IM = 1-(np.array(img))/255.
    IM = cv2.dilate(IM, dilation_kernel)

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(IM, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    return horizontal_lines_img

def vlines(img, kernel_length=20, dilation_kernel=np.ones((1,1))):
    'Find vertical lines in a kernel'
    kernel_length = int(np.array(img).shape[0]//kernel_length)

    IM = 1-(np.array(img))/255.
    IM = cv2.dilate(IM, dilation_kernel)

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(IM, vert_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp2, vert_kernel, iterations=3)
    return vertical_lines_img

def skeleton(x):
    'Skeleton of an image'
    h = cv2.dilate(hlines(x, kernel_length=1), np.array([[0,1,0],[0,1,0],[0,1,0]]).astype(np.uint8))
    v = cv2.dilate(vlines(x, kernel_length=1), np.array([[0,0,0],[1,1,1],[0,0,0]]).astype(np.uint8))
    # h, v = h[...,0], v[...,0]
    skel = (h+v)>0.2
    return (255*skel).astype(np.uint8)

def hcut(IM, debug=False):
    'Cut image into horizontal strips'
    h, w, _ = IM.shape
    im = IM[...,0]
    x = cv2.erode(im, np.array([[0,1,0],[0,1,0],[0,1,0]]).astype(np.uint8))
    h = cv2.dilate(hlines(x), np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]).astype(np.uint8))
    nzs = np.nonzero(np.mean(h, axis=1) > 0.05)[0]
    _nzs = np.nonzero(np.diff(nzs) > 1)[0]
    _nzs = _nzs - 5
    cuts = [0] + list(nzs[_nzs])+[nzs[-1]] + [im.shape[0]]
    if debug==1:
        imc = IM.copy(); h, w, _ = imc.shape
        for ix,s in enumerate(cuts[:-1]): rect(IM, (0,s,w,cuts[ix+1]))
        show(IM, sz=20); plt.show()
    return cuts

def remove_borders(IM, debug=False):
    IM = IM[...,0] if len(IM.shape)==3 else IM
    IM = IM.copy()
    h, w = IM.shape
    if h < 10 or w < 10: return IM
    IM[:1] = 255; IM[:,:1] = 255
    IM[-1:] = 255; IM[:,-1:] = 255

    if h > 50:
        IM[5] = 255; IM[:,5] = 255
        IM[-5] = 255; IM[:,-5] = 255

    if debug: show(IM, title='image with borders')
    H, W = IM.shape[:2]
    mask = np.ones(IM.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(IM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ix,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        X,Y = x+w,y+h
        if w == W and h == H: continue
        # imc = IM.copy(); rect(imc, (x,y,X,Y)); show(imc); plt.show()
        if (abs(Y-H)<1 and abs(X-W)<1) or (abs(Y-H)<1 and abs(x)<1) or\
           (abs(y)<1 and abs(X-W)<1) or (abs(y)<1 and abs(x)<1) or\
           (not cv2.contourArea(cnt)) or\
           (w < 5 and (abs(X-W) < 5 or abs(x) < 5)) or\
           (h < 5 and (abs(Y-H) < 5 or abs(y) < 5)) or\
           (w/h > 15): # one of the corners or edges
            cv2.drawContours(mask, [cnt], -1, 0, -1)
        if debug and ix<0:
            print(x,y,w,h,X,Y,Y-H,X-W)
            imc = C(IM.copy())# ; rect(imc, (x,y,X,Y))
            cv2.drawContours(imc, [cnt], -1, 255, -1)
            show(imc); plt.show()
            # show(255-mask, sz=50); plt.show()
    if debug: show(mask, title='mask'); plt.show()
    pxls = np.nonzero(255-mask)
    IM[pxls] = 255
    if debug: show(IM, title='image with no borders')
    return IM

def get_bbs(IM, kern_sz=(1,5), debug=False, fullobjectcond=False, insidecond=True):
    IM = IM[...,0] if len(IM.shape) == 3 else IM
    mask = np.ones(IM.shape[:2], dtype="uint8") * 255
    if debug: show(IM); plt.show()
    H, W = IM.shape
    dilation = 255 - cv2.erode(IM, np.ones(kern_sz), iterations = 1)
    (thresh, im) = cv2.threshold(dilation, 0, 255, 0)
    if debug: show(im)
    contours, hierarchy = cv2.findContours(im.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbs = []
    imc = np.repeat(im[...,None], 3, 2)
    def cond():
        if fullobjectcond:
            return w > 5 and h > 5 and not(w == W and h == H)
        else:
            return True
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        X,Y = x+w,y+h
        if cond():
            bbs.append((x,y,X,Y))
            if debug: cv2.rectangle(imc, (x,y), (X, Y), (0,255,0), 2)
            if debug: cv2.putText(imc, str(w), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 1)
        else:
            if debug: cv2.rectangle(imc, (x,y), (X, Y), (255,0,0), 2)
            if debug: cv2.putText(imc, str(w), (int(x),int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 1)
    # if debug: show(imc, sz=20); plt.show()
    pxls = np.nonzero(255-mask)
    IM[pxls] = 255
    # if debug: show(IM); plt.show()
    if insidecond:
        toremove = []
        for bb in bbs:
            if any([isin(bb, _bb) for _bb in bbs]): toremove.append(bb)
        for bb in toremove: bbs.remove(bb)
    if debug:
        imc = np.repeat(IM[...,None], 3, 2)
        for bb in bbs:
            rect(imc, bb)
        show(imc)

    bbs = sorted(bbs, key=lambda x: (x[1], x[0]))
    bbs = [BB(bb) for bb in bbs]
    return bbs

def isin(boxA, boxB):
    # if boxA is in boxB
    # determine the (x, y)-coordinates of the intersection rectangle
    x,y,X,Y = boxA
    a,b,A,B = boxB
    if boxA == boxB:
        return False
    if x >= a and X <= A and y >= b and Y <= B:
        return True
    return False

'Bounding Boxes in a form'
def boxes(im, vlines_sz=100, debug=False):
    imc = im.copy()
    skl = hlines(imc, dilation_kernel=np.ones((3,3))) +\
        vlines(imc, vlines_sz, dilation_kernel=np.ones((3,3)))
    imc = C(imc)
    sklc = skl.copy() > 0.1

    if debug: show(sklc)
    # --- âœ„ -----------------------
    # OLD
    if 1:
        _im = cv2.dilate(255*(sklc).astype(np.uint8), np.ones((5,1)))/255
        # _im = sklc
        hs = np.where(_im.mean(-1) > 0.5)
        sklc[hs] = 1

    # NEW
    else:
        colmean = sklc.mean(0)
        # plt.plot(colmean); plt.show()
        L = np.nonzero(np.diff(colmean) > 0.02)[0][0]+10
        R = np.nonzero(np.diff(colmean) < -0.02)[0][-1]-10
        hs = np.where(sklc[:,L] > 0); sklc[hs,:L] = 1
        hs = np.where(sklc[:,R] > 0); sklc[hs,R:] = 1
    # --- âœ„ -----------------------

    p = 10
    sklc[:p,:] = 1; sklc[-p:,:] = 1
    sklc[:,:p] = 1; sklc[:,-p:] = 1

    sklc = cv2.dilate(sklc.astype(np.uint8), np.ones((2,2)))
    # show(sklc, sz=20)
    contours, hierarchy  = cv2.findContours(sklc.astype(np.uint8),
                                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bbs = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(imc, (x,y), (x+w, y+h), (0,255,0), 2)
        if w > 1 and h > 40:
            bb = (x,y-4,x+w,y+4+h)
            bbs.append(BB(bb))

    def content_in_right_corners(im, bb):
        crop = crop_from_bb(im, bb)
        crop = remove_borders(crop)
        _crop = crop[3:-3,-10:-3]
        info = np.mean(_crop) < np.median(crop), np.round(np.mean(_crop), 2)
        # show(crop, title=info[1])
        return info

    def _merge_small_bbs(bbs):
        def merge_nearest_right(bb, bbs):
            toadd, toremove = [], []
            _bb = nearest(bb, bbs, kind='right')
            if _bb is None: return
            toadd.append(combine_bbs([bb,_bb]))
            [toremove.append(bb_) for bb_ in [bb, _bb]]
            for bb in toremove:
                try: bbs.remove(bb)
                except Exception as e: ... # print(e)
            for bb in toadd: bbs.append(bb)

        _bbs = []
        for bb in bbs:
            prop, _ = content_in_right_corners(imc, bb)
            if prop:
                _bbs.append(bb)
                merge_nearest_right(bb, bbs)
        return

    # [_merge_small_bbs(bbs) for _ in range(5)]
    bbs = sorted(bbs, key=lambda x: (x[1], x[0]))
    bbs = [BB(bb) for bb in bbs]
    if debug: show(imc, bbs=bbs, texts=range(len(bbs)), sz=40)
    # show(imc, sz=20)
    return bbs
    'usage'
    im = read(f)
    im = derotate(im, skew_angle(im[::2,::2]  > 127))
    # im = remove_margins(im)
    imc = C(im.copy())
    bbs = boxes(imc[...,0])
    for bb in bbs:
        x,y,w,h = bb
        cv2.rectangle(imc, (x,y), (x+w, y+h), (0,255,0), 2)
    # show(imc, sz=20)

def C(im):
    'make bw into 3 channels'
    if im.shape==3: return im
    else:
        return np.repeat(im[...,None], 3, 2)

def B(im,thr=180): return 255*(im > thr).astype(np.uint8)

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x,y,X,Y = boxA
    a,b,A,B = boxB

    xA = max(x,a)
    yA = max(y,b)
    xB = min(X,A)
    yB = min(Y,B)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = ((boxA[0]+boxA[2]) - boxA[0] + 1) * ((boxA[1]+boxA[3]) - boxA[1] + 1)
    boxBArea = ((boxB[0]+boxB[2]) - boxB[0] + 1) * ((boxB[1]+boxB[3]) - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def impad(im, pad=5):
    im = im[...,0] if len(im.shape) == 3 else im
    h, w = im.shape
    IM = np.ones((h+pad*2, w+pad*2))*255
    IM[pad:-pad,pad:-pad] = im
    return IM

def _merge(bbs):
    'combine two bbs into one if they have non-zero iou'
    toremove = set()
    toadd = set()
    for bb1 in bbs:
        x,y,X,Y = bb1
        for bb2 in bbs:
            a,b,A,B = bb2
            if bb1 != bb2:
                if bb_iou(bb1, bb2) > 0 and abs(y-b)<20:
                    toremove.add(bb1)
                    toremove.add(bb2)
                    c,d,C,D = min(x,a),min(y,b),max(X,A),max(Y,B)
                    toadd.add(BB((c,d,C,D)))
    for bb in toremove: bbs.remove(bb)
    for bb in toadd   : bbs.append(bb)
    bbs = [BB(bb) for bb in bbs]
    return bbs

def merge(bbs):
    for i in range(5): out = _merge(bbs)
    return out

def get_bounding_image(im):
    bbs = get_bbs(im)
    h,w = im.shape
    if len(bbs) < 1: return im, (0,0,w,h)
    xs,ys,Xs,Ys = zip(*bbs)
    x,y,X,Y = min(xs),min(ys),max(Xs),max(Ys)
    x = max(0, x-1)
    return crop_from_bb(im, (x,y,X,Y)), (x,y,X,Y)

def combine_bbs(bbs):
    xs,ys,Xs,Ys = zip(*bbs)
    x,y,X,Y = min(xs),min(ys),max(Xs),max(Ys)
    x = max(0, x)
    y = max(0, y)
    return BB((x,y,X,Y))

def nearest(bb, bbs, eps=5, kind='right'):
    'eps: how close should the y-margins be'
    bbs = [BB(bb) for bb in bbs]
    x,y,X,Y = _bb_ = BB(bb)
    if len(bbs) == 0: return
    if kind in ['left', 'right']:
        bbs = [bb for bb in bbs if abs(y-bb.y) < eps and abs(Y-bb.Y) < eps and bb != _bb_]
    elif kind in ['top', 'bottom']:
        bbs = [bb for bb in bbs if abs(x-bb.x) < eps and bb != _bb_]

    if kind == 'right':
        bbs = [bb for bb in bbs if bb.X-x > 0]
        key = lambda ix:   abs(X - bbs[ix].x)
    elif kind == 'left' :
        bbs = [bb for bb in bbs if bb.x-X < 0]
        key = lambda ix: abs(x - bbs[ix].X)
    elif kind == 'top'  :
        key = lambda ix: abs(y - bbs[ix].Y)
    elif kind =='bottom':
        key = lambda ix: abs(Y - bbs[ix].y)
    else: raise NotImplemented

    if len(bbs) == 0: return BB(0,0,0,0), -1
    ix = min(range(len(bbs)), key=key)
    return bbs[ix], ix

"DEROTATION"
from scipy.ndimage import interpolation as inter
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skew_angle(im):
    'make sure image is binary. Send a small image'
    delta = 0.5
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(im, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.format(best_angle))
    return best_angle
    'usage'
    best_angle = derotate(im[::2,::2])
    inter.rotate(im, best_angle, reshape=True, order=0, cval=1)

def derotate(im, shrinkfactor=2, angle=None):
    sf = shrinkfactor
    angle = skew_angle(im[::sf,::sf] > 127) if angle is None else angle
    return inter.rotate(im, angle, reshape=True, order=0, cval=255)

def resize(im, shape, pad=None, how=min):
    if isinstance(shape, int):
        shape = (shape, shape)
    if isinstance(shape, (tuple, np.ndarray)):
        H, W = shape
        h, w = im.shape
        f = how(H/h, W/w)        
    if isinstance(shape, float):
        f = shape
    im = cv2.resize(im, None, fx=f, fy=f)
    return im

def pad(im, shape, pad_val=255):
    if isinstance(shape, int):
        shape = (shape, shape)
    h, w = im.shape
    _im = np.ones(shape)*pad_val
    _im[:h,:w] = im
    return _im
