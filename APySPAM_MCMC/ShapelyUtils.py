'''
Author: David O'Ryan
Date: 29/11/2022

This script will have all the utility functions used regarding Shapely. This is primarily used to form a Chi Squared based on the galaxy shape.
'''

from shapely.geometry import Polygon
from shapely.ops import unary_union

import numpy as np
import cv2 as cv

import matplotlib
import matplotlib.pyplot as plt

class ShapelyUtils:
    def get_galaxy_polygon(im, sec_pos, res, init_check):

        flag = True

        pixel_values = [int(25 + (sec_pos[1] / res)), int(25 + (sec_pos[0] / res))]

        # Get Binary Cutout
        prim_cutout, prim_bin_cutout = get_cutout(im, [int(im.shape[0]/2), int(im.shape[1]/2)])
        sec_cutout, sec_bin_cutout = get_cutout(im, pixel_values)

        ## Get the Contours (Returns a nested list of all contours)
        contours_prim, _ = cv.findContours(prim_bin_cutout, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE)
        contours_sec, _ = cv.findContours(sec_bin_cutout, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE)

        if len(contours_prim) == 0 or len(contours_sec) == 0:
            return [],[],False

        prim_polygon, prim_flag = get_correct_polygon(contours_prim)
        sec_polygon, sec_flag = get_correct_polygon(contours_sec)

        if prim_flag == False or sec_flag == False:
            flag = False

        if init_check:
            plt.figure(figsize = (12,8))
            plt.imshow(np.log10(prim_cutout))
            plt.savefig('/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/test-image/test-image/prim-cutout.jpg')
            plt.close()

            plt.figure(figsize = (12,8))
            plt.imshow(np.log10(sec_cutout))
            plt.savefig('/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/test-image/test-image/sec-cutout.jpg')
            plt.close()

        return [prim_cutout, sec_cutout], [prim_polygon, sec_polygon], flag

    def calculate_polygon_flux(cutout, polygon, flag):

        if flag == 'obs':
            cutout_centered_poly = Polygon(np.asarray(polygon.exterior.xy).T + np.asarray([cutout.shape[0]/2, cutout.shape[1]/2]))
            cutout_centered_poly_arr = matplotlib.path.Path(np.asarray(cutout_centered_poly.exterior.xy).T.astype(int))
        elif flag == 'sim':
            # get index of max pixel
            centre_x, centre_y = np.where(cutout == np.max(cutout))
            cutout_centered_poly = Polygon(np.asarray(polygon.exterior.xy).T + np.asarray([centre_x[0], centre_y[0]]))
            cutout_centered_poly_arr = matplotlib.path.Path(np.asarray(cutout_centered_poly.exterior.xy).T.astype(int))
        else:
            print('Error')
            sys.exit()

        x = np.arange(0, cutout.shape[0])
        y = np.arange(0, cutout.shape[1])

        xv, yv = np.meshgrid(x,y,indexing='xy')
        points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
        mask = cutout_centered_poly_arr.contains_points(points)
        mask.shape = xv.shape

        galaxy_flux = np.sum(cutout[mask.T])

        return galaxy_flux

    def get_jaccard_dist(poly, sim_poly):
        prim_poly, sec_poly = poly
        sim_prim_poly, sim_sec_poly = sim_poly

        ## Prim Jaccard Distance
        prim_intersection = prim_poly.intersection(sim_prim_poly).area
        prim_union = unary_union([prim_poly, sim_prim_poly]).area
        prim_jaccard_dist = 1 - (prim_intersection / prim_union)

        sec_intersection = sec_poly.intersection(sim_sec_poly).area
        sec_union = unary_union([sec_poly, sim_sec_poly]).area
        sec_jaccard_dist = 1 - (sec_intersection / sec_union)

        return [prim_jaccard_dist, sec_jaccard_dist]

    
def get_cutout(im, pos):
    x_shape, y_shape = im.shape

    limits = [int(pos[0] - x_shape/5), int(pos[0] + x_shape/5), int(pos[1] - y_shape/5), int(pos[1] + y_shape/5)]

    for i in range(len(limits)):
        if limits[i] > 50:
            limits[i] = 50
        elif limits[i] < 0:
            limits[i] = 0
        
    cutout = im[limits[0]:limits[1],limits[2]:limits[3]]

    cut = np.percentile(cutout, 70)
    cutout_bin = cutout.copy()
    cutout_bin[cutout > cut] = 1
    cutout_bin[cutout <= cut] = 0

    return cutout, cutout_bin.astype('int')

def conts_to_list(contours):
    contour_list = []
    for i in range(len(contours)):
        row = contours[i][0]
        contour_list.append([row[0],row[1]])
    return contour_list

def get_correct_polygon(contours):
    length = 0
    correct_contours_nested = []
    for i in contours:
        if len(i) > length and len(i) > 3:
            correct_contours_nested = i
            length = len(i)
    
    if len(correct_contours_nested) == 0:
        return [], False

    contour_list = conts_to_list(correct_contours_nested)        

    poly = Polygon(contour_list).buffer(0.00001)
    # Need to centre the Polygon

    if poly.geom_type == 'MultiPolygon':
        poly_list = list(poly)
        area = 0
        for i in poly_list:
            if i.area > area:
                poly = i
                area = i.area

    centre = np.asarray(poly.centroid.xy)
    poly_coords = np.asarray(poly.exterior.xy).T
    centred_poly_coords = poly_coords - centre.T

    centred_poly = Polygon(centred_poly_coords)

    return centred_poly, True
