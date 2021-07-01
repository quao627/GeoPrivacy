import math

from geopy.distance import geodesic


def euclidean(pt1, pt2):
    """
        This function calculates the euclidean distance between two points.

        Parameters
        ----------
        pt1 : tuple or list
            The coordinates of the first point. Eg. (41.58, -74.35)
        pt2 : tuple or list
            The coordinates of the second point. Eg. (41.58, -74.35)

        Returns
        -------
        dist : float
            The euclidean distance between pt1 and pt2.

    """
    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    dist = math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))
    return dist


def geodesic_mile(pt1, pt2, lat_first=True):
    """
        This function calculates the mile distance between two points.

        Parameters
        ----------
        pt1 : tuple or list
            The coordinates of the first point. Eg. (41.58, -74.35)
        pt2 : tuple or list
            The coordinates of the second point. Eg. (41.58, -74.35)
        lat_first (optional) : boolean, default True
            True if the format of coordinates is (lat, lon).

        Returns
        -------
        dist : float
            The mile distance between pt1 and pt2.

    """
    if not lat_first:
        pt1 = (pt1[1], pt1[0])
        pt2 = (pt2[1], pt2[0])
    dist = geodesic(pt1, pt2).miles
    return dist


def geodesic_km(pt1, pt2, lat_first=True):
    """
        This function calculates the kilometer distance between two points.

        Parameters
        ----------
        pt1 : tuple or list
            The coordinates of the first point. Eg. (41.58, -74.35)
        pt2 : tuple or list
            The coordinates of the second point. Eg. (41.58, -74.35)
        lat_first (optional) : boolean, default True
            True if the format of coordinates is (lat, lon).

        Returns
        -------
        dist : float
            The kilometer distance between pt1 and pt2.

    """
    if not lat_first:
        pt1 = (pt1[1], pt1[0])
        pt2 = (pt2[1], pt2[0])
    dist = geodesic(pt1, pt2).km
    return dist
