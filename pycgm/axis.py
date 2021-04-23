import math
from math import pi
import numpy as np


def rotmat(x=0, y=0, z=0):
    r"""Rotation Matrix.

    This function creates and returns a rotation matrix.

    Parameters
    ----------
    x, y, z : float, optional
        Angle, which will be converted to radians, in
        each respective axis to describe the rotations.
        The default is 0 for each unspecified angle.

    Returns
    -------
    r_xyz : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import rotmat
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  , -0.01,  0.01],
    [ 0.01,  1.  , -0.01],
    [-0.01,  0.01,  1.  ]])
    >>> x = 0.5
    >>> np.around(rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.  ],
    [ 0.  ,  1.  , -0.01],
    [ 0.  ,  0.01,  1.  ]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.02],
    [ 0.  ,  1.  , -0.02],
    [-0.02,  0.02,  1.  ]])
    """
    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    r_x = [[1, 0, 0], [0, math.cos(x), math.sin(x) * -1], [0, math.sin(x), math.cos(x)]]
    r_y = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [math.sin(y) * -1, 0, math.cos(y)]]
    r_z = [[math.cos(z), math.sin(z) * -1, 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
    r_xy = np.matmul(r_x, r_y)
    r_xyz = np.matmul(r_xy, r_z)

    return r_xyz


def get_head_angle(axis_p, axis_d):
    r"""Head angle calculation function.

    This function takes in two axes and returns three angles and uses the
    inverse Euler rotation matrix in YXZ order.

    Returns the angles in degrees.

    Parameters
    ----------
    axis_p : list
        Shows the unit vector of axis_p, the position of the proximal axis.
    axis_d : list
        Shows the unit vector of axis_d, the position of the distal axis.

    Returns
    -------
    angle : list
        Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

    Notes
    -----
    :math:`\beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{x} \cdot axisP_{y})^2 + (axisD_{y} \cdot axisP_{y})^2}})`

    :math:`\alpha = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})}`

    :math:`\gamma = \arctan2{(-(axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})}`

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import get_head_angle
    >>> axis_p = [[0.04, 0.99, 0.06, 512.34],
    ...           [0.99, -0.04, -0.05, 471.15],
    ...           [-0.05,  0.07, -0.99, 124.14],
    ...           [0, 0, 0, 1]]
    >>> axis_d = [[-0.18, -0.98, -0.02, 842.14],
    ...           [ 0.71, -0.11, -0.69, 985.38],
    ...           [ 0.67, -0.14, 0.72, 412.87],
    ...           [0, 0, 0, 1]]
    >>> np.around(get_head_angle(axis_p,axis_d), 2)
    array([ 185.18,  -39.99, -190.54])
    """
    # this is the angle calculation which order is Y-X-Z
    # alpha is abdcution angle.

    ang = (
        (-1 * axis_d[2][0] * axis_p[1][0])
        + (-1 * axis_d[2][1] * axis_p[1][1])
        + (-1 * axis_d[2][2] * axis_p[1][2])
    )
    alpha = np.nan
    if -1 <= ang <= 1:
        alpha = np.arcsin(ang)

    # check the abduction angle is in the area between -pi/2 and pi/2
    # beta is flextion angle
    # gamma is rotation angle

    beta = np.arctan2(
            (axis_d[2][0] * axis_p[1][0])
            + (axis_d[2][1] * axis_p[1][1])
            + (axis_d[2][2] * axis_p[1][2]),
        np.sqrt(
            (
                axis_d[0][0] * axis_p[1][0]
                + axis_d[0][1] * axis_p[1][1]
                + axis_d[0][2] * axis_p[1][2]
            ) ** 2
            + (
                axis_d[1][0] * axis_p[1][0]
                + axis_d[1][1] * axis_p[1][1]
                + axis_d[1][2] * axis_p[1][2]
            ) ** 2
        ),
    )

    alpha = np.arctan2(
        -1 * (
            (axis_d[2][0] * axis_p[0][0])
            + (axis_d[2][1] * axis_p[0][1])
            + (axis_d[2][2] * axis_p[0][2])
        ), (
            (axis_d[2][0] * axis_p[2][0])
            + (axis_d[2][1] * axis_p[2][1])
            + (axis_d[2][2] * axis_p[2][2])
        )
    )

    gamma = np.arctan2(
        -1 * (
            (axis_d[0][0] * axis_p[1][0])
            + (axis_d[0][1] * axis_p[1][1])
            + (axis_d[0][2] * axis_p[1][2])
        ), (
            (axis_d[1][0] * axis_p[1][0])
            + (axis_d[1][1] * axis_p[1][1])
            + (axis_d[1][2] * axis_p[1][2])
        ),
    )

    alpha = 180.0 * alpha / pi
    beta = 180.0 * beta / pi
    gamma = 180.0 * gamma / pi

    beta *= -1

    if alpha < 0:
        alpha *= -1
    else:
        if 0 < alpha < 180:
            alpha = 180 + (180 - alpha)

    if gamma > 90.0:
        if gamma > 120:
            gamma = (gamma - 180) * -1
        else:
            gamma = (gamma + 180) * -1
    else:
        if gamma < 0:
            gamma = (gamma + 180) * -1
        else:
            gamma = (gamma * -1) - 180.0

    angle = [alpha, beta, gamma]

    return angle
