# ALL FUNCTION FROM PREVIOUS TASK

# libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

from math import pi, sqrt, exp, ceil


def gauss(sigma):
    kernel = list(map(
        lambda x: 1 / (sqrt(2 * pi) * sigma) * exp(-x ** 2 / (2 * sigma ** 2)),
        range(-ceil(3 * sigma), ceil(3 * sigma) + 1)
    ))
    return np.array(kernel) / sum(kernel)


def gaussdx(sigma):
    kernel = list(map(
        lambda x: -1 / (sqrt(2 * pi) * sigma ** 3) * x * exp(-x ** 2 / (2 * sigma ** 2)),
        range(-ceil(3 * sigma), ceil(3 * sigma) + 1)
    ))
    return np.array(kernel) / np.sum(np.abs(kernel))


def convert_to_2D(kernel):
    size = (len(kernel) - 1) // 2
    kernel2d = np.zeros((2 * size + 1, 2 * size + 1))
    kernel2d[size, :] = kernel  # kernel should be in the middle row of square matrix
    return kernel2d


def show_images(images, titles=None):
    for idx, im in enumerate(images):
        plt.imshow(im, cmap='gray')
        if titles is not None: plt.title(titles[idx])
        plt.show()


def flip(kernel):
    return kernel.reshape(-1)[::-1].reshape((kernel.shape))


def convolve(src, kernel):
    return cv2.filter2D(src, -1, flip(kernel))


def convolve_more_kernels(src, kernels):
    src_copy = np.copy(src)
    for kernel in kernels:
        src_copy = convolve(src_copy, kernel)
    return src_copy


def get_image_derivatives(I, sigma):
    G = convert_to_2D(gauss(sigma))
    D = convert_to_2D(gaussdx(sigma))
    Ix = convolve_more_kernels(I, [G.T, D])
    Iy = convolve_more_kernels(I, [D.T, G])
    return Ix, Iy


def get_image_2nd_derivatives(I, sigma):
    G = convert_to_2D(gauss(sigma))
    D = convert_to_2D(gaussdx(sigma))
    Ix, Iy = get_image_derivatives(I, sigma)
    Ixx = convolve_more_kernels(Ix, [D, G.T])
    Iyy = convolve_more_kernels(Iy, [D.T, G])
    Ixy = convolve_more_kernels(Ix, [D.T, G])
    return Ixx, Iyy, Ixy


from itertools import product
from scipy import ndimage


def get_hessians(src, sigma, plot_hessian=True):
    Ixx, Iyy, Ixy = get_image_2nd_derivatives(src, sigma)
    determinants = (sigma ** 4) * (Ixx * Iyy - Ixy * Ixy)
    if plot_hessian: show_images([determinants], [f'hessian sigma = {sigma}'])
    return determinants


def non_maximum_supression(src):
    src = np.copy(src)
    neighborhood = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
    neighborhood_max = ndimage.generic_filter(src, lambda x: max(x), footprint=neighborhood, mode='constant', cval=0)
    # return max value of neighborhood for each pixel
    coordinates = product(range(src.shape[0]), range(src.shape[1]))
    for coor, nei_max in zip(coordinates, neighborhood_max.reshape(-1)):
        if src[coor] <= nei_max: src[coor] = 0
    return src


def hessian_points(src, sigma, t):
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    src_gray = src_gray.astype(float)  #
    hessians = get_hessians(src_gray, sigma)
    hessians[hessians <= t] = 0  # threshold determinant
    supr_det = non_maximum_supression(hessians)
    return [(j, i) for i, j in product(range(supr_det.shape[0]), range(supr_det.shape[1])) if
            supr_det[i, j] != 0]  # get coordinates which values is not zero


def show_points(I, points):
    plt.imshow(I, cmap='gray')
    plt.plot([j for j, _ in points], [i for _, i in points], 'x', color='red')
    plt.show()


def get_harris_values(src, sigma, plot_hessian=False):
    Ix, Iy = get_image_derivatives(src, sigma)
    Gx = convert_to_2D(gauss(1.6 * sigma))
    G = convolve(Gx, Gx.T)

    # values of specific matrix indices for whole image
    C11 = convolve(Ix ** 2, G)
    C22 = convolve(Iy ** 2, G)
    C12 = convolve(Ix * Iy, G)

    det = C11 * C22 - C12 ** 2
    trace = C11 + C22
    alpha = 0.06
    values = det - alpha * trace ** 2
    if plot_hessian: show_images([values], [f'harris values for sigma = {sigma}'])
    return values


def harris_points(src, sigma, t):
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    src_gray = src_gray.astype(float)
    values = get_harris_values(src_gray, sigma)
    values[values <= t] = 0  # threshold values
    supr_det = non_maximum_supression(values)
    return [(j, i) for i, j in product(range(supr_det.shape[0]), range(supr_det.shape[1])) if
            supr_det[i, j] != 0]  # get coordinates which values is not zero


def read_data(filename):
    # reads a numpy array from a text file
    with open(filename) as f:
        s = f.read()

    return np.fromstring(s, sep=' ')


def helinger(h1, h2):
    return np.sqrt(0.5 * np.sum(np.square(np.sqrt(h1) - np.sqrt(h2))))


def find_correspodences_symetric(I1, descriptors1, I2, descriptors2):
    correspondences_1to2 = dict()
    correspondences_2to1 = dict()
    for indx, desc1 in enumerate(descriptors1):
        dist = [(i, helinger(desc1, desc2)) for i, desc2 in enumerate(descriptors2)]
        nearest_point = min(dist, key=lambda x: x[1])[0]
        correspondences_1to2[indx] = nearest_point
    for indx, desc2 in enumerate(descriptors2):
        dist = [(i, helinger(desc1, desc2)) for i, desc1 in enumerate(descriptors1)]
        nearest_point = min(dist, key=lambda x: x[1])[0]
        correspondences_2to1[indx] = nearest_point
    return [(i1, i2) for i1, i2 in correspondences_1to2.items() if correspondences_2to1[i2] == i1]


def simple_descriptors(I, pts, bins=16, radius=40, w=11):
    g = gauss(w)
    d = gaussdx(w)

    Ix = cv2.filter2D(I, cv2.CV_32F, g.T)
    Ix = cv2.filter2D(Ix, cv2.CV_32F, d)

    Iy = cv2.filter2D(I, cv2.CV_32F, g)
    Iy = cv2.filter2D(Iy, cv2.CV_32F, d.T)

    Ixx = cv2.filter2D(Ix, cv2.CV_32F, g.T)
    Ixx = cv2.filter2D(Ixx, cv2.CV_32F, d)

    Iyy = cv2.filter2D(Iy, cv2.CV_32F, g)
    Iyy = cv2.filter2D(Iyy, cv2.CV_32F, d.T)

    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    mag = np.floor(mag * ((bins - 1) / np.max(mag)))

    feat = Ixx + Iyy
    feat += abs(np.min(feat))
    feat = np.floor(feat * ((bins - 1) / np.max(feat)))

    desc = []

    for y, x in pts:
        minx = max(x - radius, 0)
        maxx = min(x + radius, I.shape[0])
        miny = max(y - radius, 0)
        maxy = min(y + radius, I.shape[1])
        r1 = mag[minx:maxx, miny:maxy].reshape(-1)
        r2 = feat[minx:maxx, miny:maxy].reshape(-1)

        a = np.zeros((bins, bins))
        for m, l in zip(r1, r2):
            a[int(m), int(l)] += 1

        a = a.reshape(-1)
        a /= np.sum(a)

        desc.append(a)

    return np.array(desc)

def find_correspodences(descriptors1, descriptors2):
    correspondences = []
    for indx, desc1 in enumerate(descriptors1): # index and descriptor of p1
        dist = [(i,helinger(desc1,desc2)) for i,desc2 in enumerate(descriptors2)] # [(indx of p2, distance to p1), ...]
        nearest_point = min(dist, key=lambda x: x[1])[0] # index of nearest point to p1
        correspondences.append((indx,nearest_point))
    return correspondences


def find_matches(I1, I2, sigma=3, threshold=1000, symetric=True):
    ps1 = harris_points(I1, sigma, threshold)
    ps2 = harris_points(I2, sigma, threshold)
    desc1 = simple_descriptors(I1, ps1)
    desc2 = simple_descriptors(I2, ps2)
    if symetric:
        corr = find_correspodences_symetric(I1, desc1, I2, desc2)
    else:
        corr = find_correspodences(desc1,desc2)
    return ps1, ps2, corr

def display_matches(im1, im2, pts1, pts2, matches):
	# NOTE: this will only work correctly for images with the same height
	# NOTE: matches should contain index pairs (i.e. first element is the index to pts1 and second for pts2)

	I = np.hstack((im1, im2))
	w = im1.shape[1]
	plt.clf()
	plt.imshow(I)

	for i, j in matches:
		p1 = pts1[int(i)]
		p2 = pts2[int(j)]
		plt.plot(p1[0], p1[1], 'bo')
		plt.plot(p2[0] + w, p2[1], 'bo')
		plt.plot([p1[0], p2[0] + w], [p1[1], p2[1]], 'r')

	plt.draw()
	plt.show()