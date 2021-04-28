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

def elbowJointCenter(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax, shoulderJC, r_elbow_width, l_elbow_width, r_wrist_width, l_wrist_width):
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
    thorax : array
        The x,y,z position of the thorax.
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
    >>> from .axis import elbowJointCenter
    >>> thorax = np.array([[ 0.09,  1., 0.01, 256.14],
    ...    [ 1.  , -0.09, -0.07, 364.3 ],
    ...    [-0.06, -9.98, -1.  , 1459.65],
    ...    [ 0.  ,  0.  , 0.  , 1.  ]])
    >>> shoulderJC = [np.array([[1., 0., 0., 429.66],
    ...   [0., 1., 0.,  275.06],
    ...   [0., 0., 1., 1453.95],
    ...   [0., 0., 0.,    1.  ]]),
    ...   np.array([[1., 0., 0., 64.51],
    ...   [0., 1., 0., 274.93],
    ...   [0., 0., 1.,1463.63],
    ...   [0., 0., 0.,   1.  ]])
    ...   ]
    >>> [np.around(arr, 2) for arr in elbowJointCenter(
    ... np.array([428.88, 270.55, 1500.73]), 
    ... np.array([68.24, 269.01, 1510.10]), 
    ... np.array([658.90, 326.07, 1285.28]), 
    ... np.array([-156.32, 335.25, 1287.39]), 
    ... np.array([776.51,495.68, 1108.38]), 
    ... np.array([830.90, 436.75, 1119.11]), 
    ... np.array([-249.28, 525.32, 1117.09]), 
    ... np.array([-311.77, 477.22, 1125.16]),
    ... thorax,
    ... shoulderJC,
    ... 74.0,
    ... 74.0,
    ... 55.0,
    ... 55.0)] #doctest: +NORMALIZE_WHITESPACE
    [array([[ 633.66,  304.95, 1256.07],
    [-129.16,  316.86, 1258.06]]), array([[[ 633.81,  303.96, 1256.07],
    [ 634.35,  305.05, 1256.79],
    [ 632.95,  304.84, 1256.77]],
    [[-129.32,  315.88, 1258.  ],
    [-128.45,  316.79, 1257.36],
    [-128.49,  316.72, 1258.78]]]), array([[ 793.32,  451.29, 1084.43],
    [-272.46,  485.79, 1091.37]])]
    """

    r_elbow_width *= -1
    mm = 7.0
    R_delta =(r_elbow_width/2.0)-mm
    L_delta =(l_elbow_width/2.0)+mm

    RWRI = [(rwra[0]+rwrb[0])/2.0,(rwra[1]+rwrb[1])/2.0,(rwra[2]+rwrb[2])/2.0]
    LWRI = [(lwra[0]+lwrb[0])/2.0,(lwra[1]+lwrb[1])/2.0,(lwra[2]+lwrb[2])/2.0]

    # make humerus axis
    tho_y_axis = thorax[1, :3]

    R_sho_mod = [(rsho[0]-R_delta*tho_y_axis[0]-relb[0]),
                (rsho[1]-R_delta*tho_y_axis[1]-relb[1]),
                (rsho[2]-R_delta*tho_y_axis[2]-relb[2])]
    L_sho_mod = [(lsho[0]+L_delta*tho_y_axis[0]-lelb[0]),
                (lsho[1]+L_delta*tho_y_axis[1]-lelb[1]),
                (lsho[2]+L_delta*tho_y_axis[2]-lelb[2])]

    # right axis
    z_axis = R_sho_mod
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    # this is reference axis
    x_axis = np.subtract(RWRI,relb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    R_axis = [x_axis,y_axis,z_axis]

    # left axis
    z_axis = np.subtract(L_sho_mod,lelb)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    # this is reference axis
    x_axis = L_sho_mod
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(z_axis,x_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    L_axis = [x_axis,y_axis,z_axis]

    RSJC = [shoulderJC[0][0][3], shoulderJC[0][1][3], shoulderJC[0][2][3]]
    LSJC = [shoulderJC[1][0][3], shoulderJC[1][1][3], shoulderJC[1][2][3]]

    # make the construction vector for finding Elbow joint center
    R_con_1 = np.subtract(RSJC,relb)
    R_con_1_div = np.linalg.norm(R_con_1)
    R_con_1 = [R_con_1[0]/R_con_1_div,R_con_1[1]/R_con_1_div,R_con_1[2]/R_con_1_div]

    R_con_2 = np.subtract(RWRI,relb)
    R_con_2_div = np.linalg.norm(R_con_2)
    R_con_2 = [R_con_2[0]/R_con_2_div,R_con_2[1]/R_con_2_div,R_con_2[2]/R_con_2_div]

    R_cons_vec = np.cross(R_con_1,R_con_2)
    R_cons_vec_div = np.linalg.norm(R_cons_vec)
    R_cons_vec = [R_cons_vec[0]/R_cons_vec_div,R_cons_vec[1]/R_cons_vec_div,R_cons_vec[2]/R_cons_vec_div]

    R_cons_vec = [R_cons_vec[0]*500+relb[0],R_cons_vec[1]*500+relb[1],R_cons_vec[2]*500+relb[2]]

    L_con_1 = np.subtract(LSJC,lelb)
    L_con_1_div = np.linalg.norm(L_con_1)
    L_con_1 = [L_con_1[0]/L_con_1_div,L_con_1[1]/L_con_1_div,L_con_1[2]/L_con_1_div]

    L_con_2 = np.subtract(LWRI,lelb)
    L_con_2_div = np.linalg.norm(L_con_2)
    L_con_2 = [L_con_2[0]/L_con_2_div,L_con_2[1]/L_con_2_div,L_con_2[2]/L_con_2_div]

    L_cons_vec = np.cross(L_con_1,L_con_2)
    L_cons_vec_div = np.linalg.norm(L_cons_vec)

    L_cons_vec = [L_cons_vec[0]/L_cons_vec_div,L_cons_vec[1]/L_cons_vec_div,L_cons_vec[2]/L_cons_vec_div]

    L_cons_vec = [L_cons_vec[0]*500+lelb[0],L_cons_vec[1]*500+lelb[1],L_cons_vec[2]*500+lelb[2]]

    REJC = find_joint_center(R_cons_vec,RSJC,relb,R_delta)
    LEJC = find_joint_center(L_cons_vec,LSJC,lelb,L_delta)


    # this is radius axis for humerus

    # right
    x_axis = np.subtract(rwra,rwrb)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    z_axis = np.subtract(REJC,RWRI)
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

    z_axis = np.subtract(LEJC,LWRI)
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

    RWJC = [RWRI[0]+r_wrist_width*R_radius[1][0],RWRI[1]+r_wrist_width*R_radius[1][1],RWRI[2]+r_wrist_width*R_radius[1][2]]
    LWJC = [LWRI[0]-l_wrist_width*L_radius[1][0],LWRI[1]-l_wrist_width*L_radius[1][1],LWRI[2]-l_wrist_width*L_radius[1][2]]

    # recombine the humerus axis

        #right

    z_axis = np.subtract(RSJC,REJC)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(RWJC,REJC)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(x_axis,z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    # attach each calulcated elbow axis to elbow joint center.
    x_axis = [x_axis[0]+REJC[0],x_axis[1]+REJC[1],x_axis[2]+REJC[2]]
    y_axis = [y_axis[0]+REJC[0],y_axis[1]+REJC[1],y_axis[2]+REJC[2]]
    z_axis = [z_axis[0]+REJC[0],z_axis[1]+REJC[1],z_axis[2]+REJC[2]]

    R_axis = [x_axis,y_axis,z_axis]

    # left

    z_axis = np.subtract(LSJC,LEJC)
    z_axis_div = np.linalg.norm(z_axis)
    z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]

    x_axis = np.subtract(LWJC,LEJC)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    y_axis = np.cross(x_axis,z_axis)
    y_axis_div = np.linalg.norm(y_axis)
    y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]

    x_axis = np.cross(y_axis,z_axis)
    x_axis_div = np.linalg.norm(x_axis)
    x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]

    # attach each calulcated elbow axis to elbow joint center.
    x_axis = [x_axis[0]+LEJC[0],x_axis[1]+LEJC[1],x_axis[2]+LEJC[2]]
    y_axis = [y_axis[0]+LEJC[0],y_axis[1]+LEJC[1],y_axis[2]+LEJC[2]]
    z_axis = [z_axis[0]+LEJC[0],z_axis[1]+LEJC[1],z_axis[2]+LEJC[2]]

    L_axis = [x_axis,y_axis,z_axis]

    axis = [R_axis,L_axis]

    origin = [REJC,LEJC]
    wrist_O = [RWJC,LWJC]

    return [origin,axis,wrist_O]