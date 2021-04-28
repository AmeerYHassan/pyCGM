import numpy as np
import math

def find_joint_center(p_a, p_b, p_c, delta):
    r"""Calculate the Joint Center.

    This function is based on the physical markers p_a, p_b, p_c
    and the resulting joint center are all on the same plane.

    Parameters
    ----------
    p_a, p_b, p_c : list
        Three markers x, y, z position of a, b, c.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.

    Returns
    -------
    joint_center : array
        Returns the joint center's x, y, z positions in a 1x3 array.

    Notes
    -----
    :math:`vec_{1} = p\_a-p\_c, \ vec_{2} = (p\_b-p\_c), \ vec_{3} = vec_{1} \times vec_{2}`

    :math:`mid = \frac{(p\_b+p\_c)}{2.0}`

    :math:`length = (p\_b - mid)`

    :math:`\theta = \arccos(\frac{delta}{vec_{2}})`

    :math:`\alpha = \cos(\theta*2), \ \beta = \sin(\theta*2)`

    :math:`u_x, u_y, u_z = vec_{3}`

    .. math::

        rot =
        \begin{bmatrix}
            \alpha+u_x^2*(1-\alpha) & u_x*u_y*(1.0-\alpha)-u_z*\beta & u_x*u_z*(1.0-\alpha)+u_y*\beta \\
            u_y*u_x*(1.0-\alpha+u_z*\beta & \alpha+u_y^2.0*(1.0-\alpha) & u_y*u_z*(1.0-\alpha)-u_x*\beta \\
            u_z*u_x*(1.0-\alpha)-u_y*\beta & u_z*u_y*(1.0-\alpha)+u_x*\beta & \alpha+u_z**2.0*(1.0-\alpha) \\
        \end{bmatrix}

    :math:`r\_vec = rot * vec_2`

    :math:`r\_vec = r\_vec * \frac{length}{norm(r\_vec)}`

    :math:`joint\_center = r\_vec + mid`

    Examples
    --------
    >>> import numpy as np
    >>> from .axis import find_joint_center
    >>> p_a = [468.14, 325.09, 673.12]
    >>> p_b = [355.90, 365.38, 940.69]
    >>> p_c = [452.35, 329.06, 524.77]
    >>> delta = 59.5
    >>> find_joint_center(p_a, p_b, p_c, delta).round(2)
    array([396.25, 347.92, 518.63])
    """

    # make the two vector using 3 markers, which is on the same plane.
    vec_1 = (p_a[0]-p_c[0], p_a[1]-p_c[1], p_a[2]-p_c[2])
    vec_2 = (p_b[0]-p_c[0], p_b[1]-p_c[1], p_b[2]-p_c[2])

    # vec_3 is cross vector of vec_1, vec_2, and then it normalized.
    vec_3 = np.cross(vec_1, vec_2)
    vec_3_div = np.linalg.norm(vec_3)
    vec_3 = [vec_3[0]/vec_3_div, vec_3[1]/vec_3_div, vec_3[2]/vec_3_div]

    mid = [(p_b[0]+p_c[0])/2.0, (p_b[1]+p_c[1])/2.0, (p_b[2]+p_c[2])/2.0]
    length = np.subtract(p_b, mid)
    length = np.linalg.norm(length)

    theta = math.acos(delta/np.linalg.norm(vec_2))

    cs_th = math.cos(theta*2)
    sn_th = math.sin(theta*2)

    u_x, u_y, u_z = vec_3

    # This rotation matrix is called Rodriques' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which
    # means three physical markers on the segment can make a plane.
    # then the orthogonal vector of the plane will be rotating axis.
    # joint center is determined by rotating the one vector of plane around rotating axis.

    rot = np.matrix([
        [cs_th+u_x**2.0*(1.0-cs_th),u_x*u_y*(1.0-cs_th)-u_z*sn_th,u_x*u_z*(1.0-cs_th)+u_y*sn_th],
        [u_y*u_x*(1.0-cs_th)+u_z*sn_th,cs_th+u_y**2.0*(1.0-cs_th),u_y*u_z*(1.0-cs_th)-u_x*sn_th],
        [u_z*u_x*(1.0-cs_th)-u_y*sn_th,u_z*u_y*(1.0-cs_th)+u_x*sn_th,cs_th+u_z**2.0*(1.0-cs_th)]
    ])

    r_vec = rot * (np.matrix(vec_2).transpose())
    r_vec = r_vec * length/np.linalg.norm(r_vec)

    r_vec = [r_vec[0,0], r_vec[1,0], r_vec[2,0]]
    joint_center = np.array([r_vec[0]+mid[0], r_vec[1]+mid[1], r_vec[2]+mid[2]])

    return joint_center

def elbow_axis(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, shoulderJC, r_elbow_width, l_elbow_width, r_wrist_width, l_wrist_width, mm):
    """Calculate the Elbow joint axis (Humerus) function.

    Takes in a dictionary of marker names to x, y, z positions, the thorax
    axis, and shoulder joint center.

    Calculates each elbow joint axis.
    
    Markers used: rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb
    Subject Measurement values used: r_elbow_width, l_elbow_width

    Parameters
    ----------
    frame
        Dictionaries of marker lists.
    shoulderJC : array
        The x,y,z position of the shoulder joint center.
    vsk : dict, optional
        A dictionary containing subject measurements.

    Returns
    -------
    origin, axis, wrist_O : array
        Returns an array containing a 2x3 array containing the right
        elbow x, y, z marker positions 1x3, and the left elbow x, y,
        z marker positions 1x3, which is followed by a 2x3x3 array containing
        right elbow x, y, z axis components (1x3x3) followed by the left x, y, z axis
        components (1x3x3) which is then followed by the right wrist joint center
        x, y, z marker positions 1x3, and the left wrist joint center x, y, z marker positions 1x3.


    Examples
    --------
    >>> import numpy as np
    >>> from .axis import elbow_axis
    >>> np.set_printoptions(suppress=True)
    >>> shoulderJC = [np.array([[1., 0., 0., 429.66],
    ...   [0., 1., 0.,  275.06],
    ...   [0., 0., 1., 1453.95],
    ...   [0., 0., 0.,    1.  ]]),
    ...   np.array([[1., 0., 0., 64.51],
    ...   [0., 1., 0., 274.93],
    ...   [0., 0., 1.,1463.63],
    ...   [0., 0., 0.,   1.  ]])
    ...   ]
    >>> [np.around(arr, 2) for arr in elbow_axis(
    ... np.array([428.88, 270.55, 1500.73]), 
    ... np.array([68.24, 269.01, 1510.10]), 
    ... np.array([658.90, 326.07, 1285.28]), 
    ... np.array([-156.32, 335.25, 1287.39]), 
    ... np.array([776.51,495.68, 1108.38]), 
    ... np.array([830.90, 436.75, 1119.11]), 
    ... np.array([-249.28, 525.32, 1117.09]), 
    ... np.array([-311.77, 477.22, 1125.16]),
    ... shoulderJC,
    ... 74.0, 74.0, 55.0, 55.0, 7.0)] #doctest: +NORMALIZE_WHITESPACE
    [array([[   0.14,   -0.99,   -0.  ,  633.66],
            [   0.69,    0.1 ,    0.72,  304.95],
            [  -0.71,   -0.1 ,    0.69, 1256.07],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.15,   -0.99,   -0.06, -129.16],
            [   0.72,   -0.07,   -0.69,  316.86],
            [   0.68,   -0.15,    0.72, 1258.06],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[[   1.  ,    0.  ,    0.  ,  793.32],
            [   0.  ,    1.  ,    0.  ,  451.29],
            [   0.  ,    0.  ,    1.  , 1084.43],
            [   0.  ,    0.  ,    0.  ,    1.  ]],
            [[   1.  ,    0.  ,    0.  , -272.46],
            [   0.  ,    1.  ,    0.  ,  485.79],
            [   0.  ,    0.  ,    1.  , 1091.37],
            [   0.  ,    0.  ,    0.  ,    1.  ]]])]
    """

    r_elbow_width *= -1
    r_delta =(r_elbow_width/2.0)-mm
    l_delta =(l_elbow_width/2.0)+mm

    rwri = [(rwra[0]+rwrb[0])/2.0,(rwra[1]+rwrb[1])/2.0,(rwra[2]+rwrb[2])/2.0]
    lwri = [(lwra[0]+lwrb[0])/2.0,(lwra[1]+lwrb[1])/2.0,(lwra[2]+lwrb[2])/2.0]

    rsjc = [shoulderJC[0][0][3], shoulderJC[0][1][3], shoulderJC[0][2][3]]
    lsjc = [shoulderJC[1][0][3], shoulderJC[1][1][3], shoulderJC[1][2][3]]

    # make the construction vector for finding Elbow joint center
    r_con_1 = np.subtract(rsjc,relb)
    r_con_1_div = np.linalg.norm(r_con_1)
    r_con_1 = [r_con_1[0]/r_con_1_div,r_con_1[1]/r_con_1_div,r_con_1[2]/r_con_1_div]

    r_con_2 = np.subtract(rwri,relb)
    r_con_2_div = np.linalg.norm(r_con_2)
    r_con_2 = [r_con_2[0]/r_con_2_div,r_con_2[1]/r_con_2_div,r_con_2[2]/r_con_2_div]

    r_cons_vec = np.cross(r_con_1,r_con_2)
    r_cons_vec_div = np.linalg.norm(r_cons_vec)
    r_cons_vec = [r_cons_vec[0]/r_cons_vec_div,r_cons_vec[1]/r_cons_vec_div,r_cons_vec[2]/r_cons_vec_div]

    r_cons_vec = [r_cons_vec[0]*500+relb[0],r_cons_vec[1]*500+relb[1],r_cons_vec[2]*500+relb[2]]

    l_con_1 = np.subtract(lsjc,lelb)
    l_con_1_div = np.linalg.norm(l_con_1)
    l_con_1 = [l_con_1[0]/l_con_1_div,l_con_1[1]/l_con_1_div,l_con_1[2]/l_con_1_div]

    l_con_2 = np.subtract(lwri,lelb)
    l_con_2_div = np.linalg.norm(l_con_2)
    l_con_2 = [l_con_2[0]/l_con_2_div,l_con_2[1]/l_con_2_div,l_con_2[2]/l_con_2_div]

    l_cons_vec = np.cross(l_con_1,l_con_2)
    l_cons_vec_div = np.linalg.norm(l_cons_vec)

    l_cons_vec = [l_cons_vec[0]/l_cons_vec_div,l_cons_vec[1]/l_cons_vec_div,l_cons_vec[2]/l_cons_vec_div]

    l_cons_vec = [l_cons_vec[0]*500+lelb[0],l_cons_vec[1]*500+lelb[1],l_cons_vec[2]*500+lelb[2]]

    rejc = find_joint_center(r_cons_vec,rsjc,relb,r_delta)
    lejc = find_joint_center(l_cons_vec,lsjc,lelb,l_delta)

    # this is radius axis for humerus
    # right
    x_axis = np.subtract(rwra,rwrb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    z_axis = np.subtract(rejc,rwri)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    R_radius = [x_axis,y_axis,z_axis]

    # left
    x_axis = np.subtract(lwra,lwrb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    z_axis = np.subtract(lejc,lwri)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    L_radius = [x_axis,y_axis,z_axis]

    # calculate wrist joint center for humerus
    r_wrist_width = (r_wrist_width / 2.0 + mm )
    l_wrist_width = (l_wrist_width / 2.0 + mm )

    rwjc = [rwri[0]+r_wrist_width*R_radius[1][0],rwri[1]+r_wrist_width*R_radius[1][1],rwri[2]+r_wrist_width*R_radius[1][2]]
    lwjc = [lwri[0]-l_wrist_width*L_radius[1][0],lwri[1]-l_wrist_width*L_radius[1][1],lwri[2]-l_wrist_width*L_radius[1][2]]

    # recombine the humerus axis
    # right
    z_axis = np.subtract(rsjc,rejc)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(rwjc,rejc)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(x_axis,z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    r_axis = np.zeros((4, 4))
    r_axis[3, 3] = 1.0
    r_axis[0, :3] = x_axis
    r_axis[1, :3] = y_axis
    r_axis[2, :3] = z_axis
    r_axis[:3, 3] = rejc

    # left
    z_axis = np.subtract(lsjc,lejc)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(lwjc,lejc)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(x_axis,z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    l_axis = np.zeros((4, 4))
    l_axis[3, 3] = 1.0
    l_axis[0, :3] = x_axis
    l_axis[1, :3] = y_axis
    l_axis[2, :3] = z_axis
    l_axis[:3, 3] = lejc

    r_wri_origin = np.identity(4)
    r_wri_origin[:3, 3] = rwjc

    l_wri_origin = np.identity(4)
    l_wri_origin[:3, 3] = lwjc

    return [r_axis, l_axis, np.array([r_wri_origin, l_wri_origin])]