import numpy as np

def again_organize(points, n):
    new_points = np.zeros(shape=points.shape)
    
    # Highest and lowest point
    corner_len = 4*(n-1)
    new_points[0] = points[0]
    new_points[-1] = points[int(corner_len/2)]
    
    rows = 2*n-3
    columns = 2*n-1
    inner_columns = columns-2
    new_row_len = corner_len
    
    # Other corner points
    # Left corner
    start = 1
    for i in range(rows):
        new_points[start+i*(columns+inner_columns)] = points[corner_len-1-i]
        
    # Right corner
    start += columns-1
    for i in range(rows):
        new_points[start+i*(columns+inner_columns)] = points[1+i]
        
    # Points above origo
    ind = corner_len
    start = 2
    for r in range(rows):
        for c in range(inner_columns):
            new_points[start+c+r*new_row_len] = points[ind]
            ind += 1
    
    # Points below origo
    start += columns-1
    for r in range(rows):
        for c in range(inner_columns):
            new_points[start+c+r*new_row_len] = points[ind+(r+1)*inner_columns-1-c]
            
    return new_points


def ball(p=2, n=50, filename=None, y_param=1.2, x_param=1.2):
    vertices = make_pnorm_points(p, n, y_param, x_param)
    faces = make_faces(n)
    
    if filename is not None:
        ballmesh = make_mesh(vertices, faces)
        ballmesh.save(filename)
    
    return vertices, faces


def calc_row(x, y, p):
    z = give_z_pnorm(x, y, p)
    row = np.array([x, y, z])
    return row


def calc_y_values(x, p, n, mass_y_parameter):
    if(x == 1):
        y_values = np.zeros(1)
    else:
        pre_values = np.array([(i/(n-1))**(mass_y_parameter-i/(n-1)) for i in range(n)])
        y_values = (1-x**p)**(1/p)*pre_values    
    return y_values


def change_sign(array, axis):
    new_arr = np.copy(array)
    for i in range(len(array)):
        new_arr[i, axis] *= -1
    return new_arr


def find_corners(points, n):
    new_array = np.zeros(shape=(4*(n-1),3))
    
    # Highest point x=1
    new_array[0] = points[n**2-n]    
    
    # Right corner points
    # Upper side
    mid_ind = 2*(n**2-n)
    for i in range(n-1):
        new_array[i+1] = points[mid_ind+(n-2-i)*(n-1)]
    # Lower side
    for i in range(n-2):
        new_array[i+n] = points[mid_ind+(n-1+i)*(n-1)]
    
    # lowest point x=-1
    low_ind = 2*n**2-3*n+1
    new_array[2*n-2] = points[low_ind]
    
    # Left corner points
    # Lower side
    mid_ind = n-1
    for i in range(n-2):
        new_array[2*n-1+i] = points[low_ind-1-i*n]
    # Upper side
    for i in range(n-1):
        new_array[3*n-3+i] = points[mid_ind+n*i]
    
    return new_array


def find_inner(points, n):
    array = np.zeros(shape=(4*n**2-12*n+9,3))
        
    left_up_ind = n**2-n-2
    right_up_ind = 3*n**2-6*n+4
    left_low_ind = n**2-1
    right_low_ind = 3*n**2-5*n+3
    
    row_len = 2*n-3
    
    # Left up
    start = 0
    for r in range(n-1):
        for c in range(n-1):
            array[start+c+r*row_len] = points[left_up_ind-c-r*n]
            
    # Right up
    start += n-1
    for r in range(n-1):
        for c in range(n-2):
            array[start+c+r*row_len] = points[right_up_ind+c-r*(n-1)]
            
    # Left down
    start = row_len*(n-1)
    for r in range(n-2):
        for c in range(n-1):
            array[start+c+r*row_len] = points[left_low_ind-c+r*n]
            
    # Right down
    start += n-1
    for r in range(n-2):
        for c in range(n-2):
            array[start+c+r*row_len] = points[right_low_ind+c+r*(n-1)]
            
    return array


def give_z_pnorm(x, y, p):
    value = 1-x**p-y**p
    if(value < 0):
        value = 0
    return np.power(value, 1/p)


def make_faces(n):
    face_len = 8*(n-1)*(2*n-3)
    faces = np.zeros(shape=(face_len,3), dtype=np.int64)
    
    lpi = 8*n**2-20*n+13    # last point ind
    f_num = 4*(n-1)    #  Half of number of faces in row
    rows = 2*n-3
    
    # Highest point triangles
    for i in range(f_num-1):
        faces[i] = np.array([0, i+1, i+2])
    faces[f_num-1] = np.array([0, 1, f_num])
    
    # Lowest point triangles
    for i in range(f_num-1):
        faces[-i-1] = np.array([lpi, lpi-i-1, lpi-i-2])
    faces[-f_num] = np.array([lpi, lpi-1, lpi-f_num])
    
    # Inner triangles
    ind = f_num
    for r in range(rows-1):
        for i in range(1,f_num):
            this_row = r*f_num
            next_row = (r+1)*f_num
            
            faces[ind] = np.array([i+this_row, i+next_row, i+1+next_row])
            ind += 1
            faces[ind] = np.array([i+this_row, i+1+this_row, i+1+next_row])
            ind += 1
            
    # Triangles uniting left and right corners
    for r in range(rows-1):
        this_end = (r+1)*f_num
        next_end = (r+2)*f_num
        
        faces[ind] =np.array([this_end, this_end+1, this_end-(f_num-1)])
        ind += 1
        faces[ind] =np.array([this_end, this_end+1, next_end])
        ind += 1
        
    return faces


from stl import mesh

def make_mesh(vertices, faces):
    ball_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            ball_mesh.vectors[i][j] = vertices[f[j],:]
    return ball_mesh

    
def make_pnorm_corner_points(p, n, mass_y_parameter=1.2, mass_x_parameter=1.4):
    x_values = np.array([(i/(n-1))**(mass_x_parameter-i/(n-1)) for i in range(n)])
    n_rows = n**2 - (n-1)
    result_array = np.zeros(shape=(n_rows,3))
    result_ind = 0
    for i in range(n):
        x = x_values[i]
        y_values = calc_y_values(x, p, n, mass_y_parameter)
        for j in range(len(y_values)):
            result_array[result_ind] = calc_row(x, y_values[j], p)
            result_ind += 1
    return result_array


def make_pnorm_points(p, n, mass_y_parameter=1.2, mass_x_parameter=1.4):
    
    if(n < 2):
        raise ValueError("Luvun n täytyy olla suurempaa kuin yksi. Nyt se oli {}.".format(n))
    if(mass_y_parameter < 1 or mass_x_parameter < 1):
        raise ValueError("Massa-parametrien täytyy olla >= 1.")
    
    final_array = make_pnorm_corner_points(p, n, mass_y_parameter, mass_x_parameter)
    
    # Double points once
    turn_axis = change_sign(final_array, axis=0)
    turn_axis = turn_axis[n:]    # Poistetaan tuplatut pisteet
    final_array = np.append(final_array, turn_axis, axis=0)
    
    # Twice
    turn_axis = change_sign(final_array, axis=1)
    turn_axis = remove_double_points_y(turn_axis, n)
    final_array = np.append(final_array, turn_axis, axis=0)
    
    # Third time
    turn_axis = change_sign(final_array, axis=2)
    turn_axis = find_inner(turn_axis, n)
    
    final_array = reorganize_points(final_array, n)
    final_array = np.append(final_array, turn_axis, axis=0)
    final_array = again_organize(final_array, n)
    
    return final_array
    

def remove_double_points_y(turned_array, n):
    mask = np.ones(len(turned_array), dtype=bool)
    
    # Upper middlepoints
    for i in range(n):
        mask[i*n] = 0
    
    # Lower middlepoints
    for i in range(n,2*n-1):
        mask[(i-1)*n+1] = 0
    
    return turned_array[mask]


    # Useless but fancy
def remove_double_points_z(turned_array, n):
    mask = np.ones(len(turned_array), dtype=bool)
    
    # Vasemman puoliskon yläpisteet
    for i in range(1,n):
        mask[i*n-1] = 0
    mask[(n-1)*n] = 0
    
    # Vasemman puoliskon alapisteet
    for i in range(n-2):
        mask[n**2+i*n] = 0
    mask[n**2+(n-3)*n+1] = 0
    
    # Oikean puoliskon pisteet. Helppoa koska muodostavat suorakulmion muotoa (2n-3)x(n-1).
    for i in range(2*n-3):
        mask[2*(n**2-n)+i*(n-1)] = 0
    
    return turned_array[mask]


    # After this change first points are corner points where z=0 starting from x=1 which is highest point.
    # After that inner points starting from upper left corner and goes left to right row by row.
def reorganize_points(points, n):
    corners = find_corners(points, n)
    new_array = np.append(corners, find_inner(points, n), axis=0)
    return new_array

