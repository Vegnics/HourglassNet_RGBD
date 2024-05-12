import math as m
from typing import Tuple,List
import tensorflow as tf
import cv2
import numpy as np

@tf.function
def tf_load_image(filename: tf.Tensor, channels: int = 3, **kwargs) -> tf.Tensor:
    """Load image from filename

    Args:
        filename (tf.Tensor): string tensor containing the image path to read
        channels (int, optional): Number of image channels. Defaults to 3.

    Returns:
        tf.Tensor: Image Tensor of shape [HEIGHT, WIDTH, channels]
    """
    print("FILENAME:::",filename)
    fname = tf.io.read_file(filename)
    #print("FILENAME:::",fname)
    return tf.io.decode_image(fname, channels=channels)


@tf.function
def tf_stack(tensor: tf.Tensor, stacks: int = 1) -> tf.Tensor:
    """Stack copies of tensor on each other

    Notes:
        This function will augment the dimensionality of the tensor.
        For a 2D Tensor of shape [HEIGHT, WIDTH] the output will
        be a tensor of 3D tensor of shape [stacks, HEIGHT, WIDTH]

    Args:
        tensor (tf.Tensor): Tensor to stacks
        stacks (int, optional): Number of stacks to use. Defaults to 1.

    Returns:
        tf.Tensor: Stacked tensor of DIM = DIM(tensor) + 1
    """
    return tf.stack([tensor] * stacks, axis=0)


@tf.function
def tf_reshape_slice(tensor: tf.Tensor, shape=3, **kwargs) -> tf.Tensor:
    """Reshape 1D to 2D tensor

    Args:
        tensor (tf.Tensor): tensor to reshape
        shape (int, optional): Shape of the 2nd dimension. Defaults to 3.

    Returns:
        tf.Tensor: 2D reshaped tensor
    """
    return tf.reshape(tensor, shape=(-1, shape))


@tf.function
def tf_resize_tensor(tensor: tf.Tensor, size: int) -> tf.Tensor:
    """Apply tensor square resizing with nearest neighbor image interpolation

    Args:
        tensor (tf.Tensor): tensor to reshape
        size (int): output size

    Returns:
        tf.Tensor: resized tensor
    """
    return tf.image.resize(tensor, size=[size, size], method="bilinear") # PREVIOSLY nearest

@tf.function
def tf_rotate_tensor_OLD(tensor: tf.Tensor, angle: tf.Tensor ,tvec: tf.Tensor) -> tf.Tensor:
    N = tensor.shape[0]
    K = tf.range(0,N,1)
    # Compute the rotation matrix and translation vector
    center = tf.constant([N/2,N/2],dtype=tf.float32)
    angle = tf.cast(angle,dtype=tf.float32)*np.pi/180.0
    alpha = tf.math.cos(angle)
    beta = tf.math.sin(angle)
    t = tf.convert_to_tensor([[1.0-alpha,-beta],[beta,1.0-alpha]])@tf.reshape(center,shape=(-1,1))
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])
    R_inv = tf.linalg.inv(R)

    # Generate pairs of pixel positions
    I,J = tf.meshgrid(K,K,indexing="ij")
    I = tf.expand_dims(I,axis=2)
    J = tf.expand_dims(J,axis=2)
    indexes = tf.concat([I,J],axis=2)
    indexes = tf.reshape(indexes,shape=[-1,2])
    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),tf.dtypes.float32)

    # Rotate the original pixel positions
    r_indexes = R@indexes + t
    # Obtain the new pixel positions whose coordinates are within the image
    mask_min = tf.math.greater_equal(r_indexes,0.0)
    mask_max = tf.math.less(r_indexes,1.0*N)
    r_mask = mask_min[0,:]&mask_min[1,:]&mask_max[0,:]&mask_max[1,:]
    r_indexes = tf.floor(r_indexes)
    r_indexes = tf.transpose(r_indexes,perm=[1,0])
    r_indexes = tf.boolean_mask(r_indexes,r_mask)
    
    # Reverse the rotation over the selected pixel positions to sample the image intensity values.
    r_indexes = tf.transpose(r_indexes,perm=[1,0])
    inv_ridxs = R_inv@(r_indexes - t)
    # Obtain the original pixel positions where the original image will be sampled
    imask_min = tf.math.greater_equal(inv_ridxs,0.0)
    imask_max = tf.math.less(inv_ridxs,1.0*N)
    ir_mask = imask_min[0,:]&imask_min[1,:]&imask_max[0,:]&imask_max[1,:]
    inv_ridxs = tf.floor(inv_ridxs)
    
    # The definite original pixel position coordinates 
    inv_ridxs = tf.transpose(inv_ridxs,perm=[1,0])
    inv_ridxs = tf.cast(tf.boolean_mask(inv_ridxs,ir_mask),dtype=tf.dtypes.int32) # Px2

    # The definite rotated pixel position coordinates
    r_indexes = tf.transpose(r_indexes,perm=[1,0])
    r_indexes = tf.cast(tf.boolean_mask(r_indexes,ir_mask),dtype=tf.dtypes.int32) # Px2
    
    # Sample the pixels from the original data
    # inv_ridxs contains just a few pixel positions
    pixels_original = tf.gather_nd(tensor,inv_ridxs)
    
    # Create a new tensor with the sampled data located at the rotated positions
    img_rotated = tf.zeros_like(tensor) 
    img_rotated = tf.tensor_scatter_nd_update(img_rotated, r_indexes, pixels_original)
    _img_rotated = tf.reshape(img_rotated,shape=[-1,3])
    zero_mask = tf.equal(_img_rotated,0)
    zero_mask = zero_mask[:,0]&zero_mask[:,1]&zero_mask[:,2]

    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),tf.dtypes.int32)
    zero_idxs = tf.boolean_mask(indexes,zero_mask)
    filled_vals = tf.map_fn(lambda coord:tf_get_nearest_neighbor(coord,
                                                                 r_indexes),
                            elems=zero_idxs,
                            dtype=tf.dtypes.int32,
                            parallel_iterations=100
    )
    #print(filled_vals)
    #img_rotated = tf.tensor_scatter_nd_update(img_rotated, zero_idxs, filled_vals)
    # Get indexes from black dots
    # apply tf_get_nearest_neighbor to each black dot -> Tensor of values
    # fill the black dots with the tensor of values
    return (img_rotated,filled_vals)


@tf.function
#def tf_rotate_tensor(tensor: tf.Tensor, angle: tf.Tensor,scale: tf.Tensor,input_size: int=256) -> tf.Tensor:
def tf_rotate_tensor(tensor: tf.Tensor,tshape:tf.Tensor, angle: tf.Tensor,scale: tf.Tensor,center: tf.Tensor) -> tf.Tensor:
    #N = float(input_size)
    #N = tensor.shape[0]
    #M = tensor.shape[1]
    N = tshape[0]
    M = tshape[1]
    #K = tf.range(0,N,1)
    Ki = tf.range(0,N,1)
    Kj = tf.range(0,M,1)
    # Compute the rotation matrix and translation vector
    #center = tf.constant([N/2,N/2],dtype=tf.float32)
    center = tf.cast(center,dtype=tf.float64)
    _center2 = tf.reshape(tf.cast(tshape[0:1],dtype=tf.float64)/2.0,[-1,1])
    angle = tf.cast(angle,dtype=tf.float64)*np.pi/180.0
    alpha = tf.cast(scale,dtype=tf.float64)*tf.math.cos(angle)
    beta = -1.0*tf.cast(scale,dtype=tf.float64)*tf.math.sin(angle)
    t = tf.convert_to_tensor([[1.0-alpha,-beta],[beta,1.0-alpha]])@tf.reshape(center,shape=(-1,1))
    #R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])
    #R_inv = tf.linalg.inv(R)

    # Generate pairs of pixel positions for the rotated image
    #I,J = tf.meshgrid(K,K,indexing="ij")
    I,J = tf.meshgrid(Ki,Kj,indexing="ij") #("yx")
    I = tf.expand_dims(I,axis=2) #Y
    J = tf.expand_dims(J,axis=2) #X
    indexes = tf.concat([I,J],axis=2) # (Y,X)s
    indexes = tf.reshape(indexes,shape=[-1,2]) #Nx2
    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),tf.dtypes.float64) #2xN
    _center = tf.reshape([center[1],center[0]],[-1,1])
    # Generate indexes from the original image (regarded as the inverse rotation)
    inv_indexes = R@(indexes - _center)+_center
    #inv_indexes_x = tf.transpose(tf.clip_by_value(inv_indexes,
    #                                            0.0,
    #                                            1.0*(tf.cast(N,dtype=tf.float32)-1.0)),
    #                                            perm=[1,0])
    inv_indexes_col = tf.clip_by_value(inv_indexes[1],0.0,1.0*(tf.cast(M,dtype=tf.float64)-1.0))
    inv_indexes_row = tf.clip_by_value(inv_indexes[0],0.0,1.0*(tf.cast(N,dtype=tf.float64)-1.0))
    
    inv_indexes_col = tf.expand_dims(inv_indexes_col,axis=0)
    inv_indexes_row = tf.expand_dims(inv_indexes_row,axis=0)

    inv_indexes = tf.transpose(tf.concat([inv_indexes_row,inv_indexes_col],axis=0),perm=[1,0])
    inv_indexes = tf.cast(inv_indexes,dtype=tf.dtypes.int32)
    pixels_original = tf.gather_nd(tensor,inv_indexes)

    # Fill the whole rotated image with the data sampled from the inverse-rotated image (original)
    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),tf.dtypes.int32)
    img_rotated = tf.zeros_like(tensor) 
    img_rotated = tf.tensor_scatter_nd_update(img_rotated, indexes, pixels_original)
    return img_rotated

def tf_get_nearest_neighbor(coordinate: tf.Tensor, indexes: tf.Tensor):#, values: tf.Tensor):
    """
    Args:
    coordinate: A tensor representing a coordinate (2,)
    indexes: Tensor with image coords (P,2)
    values: Tensor with the values sampled from an image (P,3) 
    """
    coord = tf.expand_dims(coordinate,axis=0)
    error = tf.cast(coord-indexes,dtype=tf.dtypes.float32)
    dist = tf.norm(error,axis=-1)
    neighbor = tf.argmin(dist)
    return tf.cast(neighbor,dtype=tf.dtypes.int32)

@tf.function
#def tf_rotate_coords(coordinates: tf.Tensor, angle: tf.Tensor,scale: tf.Tensor, input_size: int = 256) -> tf.Tensor:
def tf_rotate_coords(coordinates: tf.Tensor,tshape:tf.Tensor, angle: tf.Tensor,scale: tf.Tensor) -> tf.Tensor:
    #N = float(input_size)
    # Compute the rotation matrix and translation vector
    #center = tf.constant([N/2,N/2],dtype=tf.float32)
    _center2 = tf.reshape(tf.cast(tshape[0:1],dtype=tf.float64)/2.0,[-1,1])
    center = tf.floor(0.5*tf.cast(coordinates[2]+coordinates[3],dtype=tf.float64))
    angle = tf.cast(angle,dtype=tf.float64)*np.pi/180.0
    alpha = tf.cast(scale,dtype=tf.float64)*tf.math.cos(angle)
    beta = -1.0*tf.cast(scale,dtype=tf.float64)*tf.math.sin(angle)
    t = tf.convert_to_tensor([[1.0-alpha,-beta],[beta,1.0-alpha]])@tf.reshape(center,shape=(-1,1))
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])  
    # Compute the rotated coordinates
    _coordinates = tf.transpose(coordinates,perm=[1,0])
    _center = tf.reshape(center,shape=(-1,1))
    rcoordinates = R@(tf.cast(_coordinates,dtype=tf.dtypes.float64)-_center)+_center#-diff

    return tf.cast(tf.transpose(rcoordinates,perm=[1,0]),dtype=tf.dtypes.float32)

def tf_rotate_norm_coords(coordinates: tf.Tensor, angle: tf.Tensor,scale: tf.Tensor) -> tf.Tensor:
    # Compute the rotation matrix and translation vector
    #coordinates = tf.cast(coordinates,dtype=tf.float64)
    scale = tf.cast(scale,dtype=tf.float64)
    center = tf.constant([1/2,1/2],dtype=tf.float64)
    angle = tf.cast(angle,dtype=tf.float64)*np.pi/180.0
    alpha = scale*tf.math.cos(-angle)
    beta = scale*tf.math.sin(-angle)
    t = tf.convert_to_tensor([[1.0-alpha,-beta],[beta,1.0-alpha]])@tf.reshape(center,shape=(-1,1))
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])

    # Compute the rotated coordinates
    _coordinates = tf.transpose(coordinates,perm=[1,0])
    rcoordinates = R@tf.cast(_coordinates,dtype=tf.dtypes.float64)+t
    rcoordinates = tf.cast(tf.clip_by_value(rcoordinates,0.0,1.0),dtype=tf.float64)
    return tf.transpose(rcoordinates,perm=[1,0])


@tf.function
def tf_squarify_coordinates_scale(coordinates: tf.Tensor, img_shape: tf.Tensor,
    bbox_factor=1.0,
    scale: float = 1.0) -> tf.Tensor:
    bbox = tf_expand_bbox(
        tf_compute_bbox(coordinates),
        img_shape,#tf.shape(image),
        bbox_factor=bbox_factor*scale,
    )
    # Get Padding
    # Once the bbox is computed we compute
    # how much V/H padding should be applied
    # Padding is necessary to conserve proportions
    # when resizing
    padding = tf_compute_padding_from_bbox(bbox)
    coordinates = coordinates - (bbox[0] - padding)
    return coordinates

@tf.function
def tf_squarify_image_scale(image: tf.Tensor,
    coordinates: tf.Tensor,
    bbox_factor=1.0,
    scale: float = 1.0) -> tf.Tensor:

    bbox = tf_expand_bbox(
        tf_compute_bbox(coordinates),
        tf.shape(image),
        bbox_factor=bbox_factor*scale,
    )
    # Get Padding
    # Once the bbox is computed we compute
    # how much V/H padding should be applied
    # Padding is necessary to conserve proportions
    # when resizing
    padding = tf_compute_padding_from_bbox(bbox)
    image = tf.pad(
        image[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0], :],
        paddings=tf_generate_padding_tensor(padding),
    )
    return image



@tf.function
def tf_compute_padding_from_bbox(bbox: tf.Tensor) -> tf.Tensor:
    """Given a bounding box tensor compute the padding needed to make a square bbox

    Notes:
        `bbox` is a 2x2 Tensor [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]

    Args:
        bbox (tf.Tensor): bounding box tensor

    Returns:
        tf.Tensor: tensor of shape (2,) containing [width paddding, height padding]
    """
    size = bbox[1] - bbox[0]
    width, height = size[0], size[1]
    # Compute Padding
    height_padding = tf.math.maximum(tf.constant(0, dtype=tf.int32), width - height)
    width_padding = tf.math.maximum(tf.constant(0, dtype=tf.int32), height - width)
    return tf.reshape([width_padding // 2, height_padding // 2], shape=(2,))


@tf.function
def tf_generate_padding_tensor(padding: tf.Tensor) -> tf.Tensor:
    """Given a Width X Height padding compute a tensor to apply `tf.pad` function

    Notes:
        `padding` argument must be consistent with `tf_compute_padding_from_bbox` output

    Args:
        padding (tf.Tensor): padding tensor

    Returns:
        tf.Tensor: tensor ready to be used with `tf.pad`
    """
    width_padding, height_padding = padding[0], padding[1]
    padding_tensor = [
        [height_padding, height_padding],
        [width_padding, width_padding],
        [0, 0],
    ]
    return padding_tensor


@tf.function
def tf_compute_bbox(coordinates: tf.Tensor, **kwargs) -> tf.Tensor:
    """From a 2D coordinates tensor compute the bounding box

    Args:
        coordinates (tf.Tensor): Joint coordinates 2D tensor

    Returns:
        tf.Tensor: Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    Xs = coordinates[:, 0]
    Ys = coordinates[:, 1]
    maxx, minx = tf.reduce_max(Xs), tf.reduce_min(Xs)
    maxy, miny = tf.reduce_max(Ys), tf.reduce_min(Ys)
    return tf_reshape_slice([minx, miny, maxx, maxy], shape=2, **kwargs)

def tf_compute_bbox_bc(coordinates: tf.Tensor, imgshape:tf.Tensor, **kwargs) -> tf.Tensor:
    """From a 2D coordinates tensor compute the bounding box (body centered)

    Args:
        coordinates (tf.Tensor): Joint coordinates 2D tensor

    Returns:
        tf.Tensor: Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    bcenter = 0.5*tf.cast(coordinates[2]+coordinates[3],dtype=tf.float32)
    dists = tf.abs(tf.cast(coordinates,tf.float32)-bcenter) 
    dxs = dists[:, 0]
    dys = dists[:, 1]
    maxx = tf.reduce_max(dxs)
    maxy = tf.reduce_max(dys)
    """
    bminx = tf.cast(bcenter[0]-maxx,tf.int32)
    bmaxx = tf.cast(bcenter[0]+maxx,tf.int32)
    bminy = tf.cast(bcenter[1]-maxy,tf.int32)
    bmaxy = tf.cast(bcenter[1]+maxy,tf.int32)
    print(bminx>0,bmaxx>0,bminy>0,bmaxy>0)
    """
    bminx = tf.cast(tf.maximum(bcenter[0]-maxx,0.0),tf.int32)
    bmaxx = tf.cast(tf.minimum(bcenter[0]+maxx,tf.cast(imgshape[1]-1,tf.float32)),tf.int32)
    bminy = tf.cast(tf.maximum(bcenter[1]-maxy,0.0),tf.int32)
    bmaxy = tf.cast(tf.minimum(bcenter[1]+maxy,tf.cast(imgshape[0]-1,tf.float32)),tf.int32)
    return tf_reshape_slice([bminx, bminy, bmaxx, bmaxy], shape=2, **kwargs)



@tf.function
def tf_expand_bbox(
    bbox: tf.Tensor, image_shape: tf.Tensor, bbox_factor: float = 1.0, **kwargs
) -> tf.Tensor:
    """Expand a bounding box area by a given factor

    Args:
        bbox (tf.Tensor): Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
        image_shape (tf.Tensor): Image shape Tensor as [Height, Width, Channels]
        bbox_factor (float, optional): Expansion factor. Defaults to 1.0.

    Returns:
        tf.Tensor: Expanded bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    # Unpack BBox
    top_left = bbox[0]
    top_left_x, top_left_y = tf.cast(top_left[0], dtype=tf.float64), tf.cast(
        top_left[1], dtype=tf.float64
    )
    bottom_right = bbox[1]
    bottom_right_x, bottom_right_y = tf.cast(
        bottom_right[0], dtype=tf.float64
    ), tf.cast(bottom_right[1], dtype=tf.float64)
    # Compute Bbox H/W
    height, width = bottom_right_y - top_left_y, bottom_right_x - top_left_x
    # Increase BBox Size
    new_tl_x = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_x - width * (bbox_factor - 1.0)
    )
    new_tl_y = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_y - height * (bbox_factor - 1.0)
    )
    new_br_x = tf.math.minimum(
        tf.cast(image_shape[1] - 1, dtype=tf.float64),
        bottom_right_x + width * (bbox_factor - 1.0),
    )
    new_br_y = tf.math.minimum(
        tf.cast(image_shape[0] - 1, dtype=tf.float64),
        bottom_right_y + height * (bbox_factor - 1.0),
    )

    new_tl_x = tf.math.floor(new_tl_x)
    new_tl_y = tf.math.floor(new_tl_y)
    new_br_x = tf.math.floor(new_br_x)
    new_br_y = tf.math.floor(new_br_y)
    #new_w = new_br_x -new_tl_x
    #new_h = 
    return tf.cast(
        tf_reshape_slice([new_tl_x, new_tl_y, new_br_x, new_br_y], shape=2, **kwargs),
        dtype=tf.int32,
    )

@tf.function
def tf_expand_bbox_squared(
    bbox: tf.Tensor, image_shape: tf.Tensor, bbox_factor: float = 1.0, **kwargs
) -> tf.Tensor:
    """Expand a bounding box area by a given factor

    Args:
        bbox (tf.Tensor): Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
        image_shape (tf.Tensor): Image shape Tensor as [Height, Width, Channels]
        bbox_factor (float, optional): Expansion factor. Defaults to 1.0.

    Returns:
        tf.Tensor: Expanded bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    bbox_factor = tf.cast(bbox_factor,tf.float64)
    # Unpack BBox
    top_left = bbox[0]
    top_left_x, top_left_y = tf.cast(top_left[0], dtype=tf.float64), tf.cast(
        top_left[1], dtype=tf.float64
    )
    bottom_right = bbox[1]
    bottom_right_x, bottom_right_y = tf.cast(
        bottom_right[0], dtype=tf.float64
    ), tf.cast(bottom_right[1], dtype=tf.float64)
    # Compute Bbox H/W
    height, width = bottom_right_y - top_left_y, bottom_right_x - top_left_x
    # Increase BBox Size
    new_tl_x = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_x - width * (bbox_factor - 1.0)
    )
    new_tl_y = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_y - height * (bbox_factor - 1.0)
    )
    new_br_x = tf.math.minimum(
        tf.cast(image_shape[1] - 1, dtype=tf.float64),
        bottom_right_x + width * (bbox_factor - 1.0),
    )
    new_br_y = tf.math.minimum(
        tf.cast(image_shape[0] - 1, dtype=tf.float64),
        bottom_right_y + height * (bbox_factor - 1.0),
    )

    nwidth,nheight = new_br_x-new_tl_x,new_br_y-new_tl_y
    N = tf.math.maximum(nwidth,nheight)
    addW = tf.math.maximum(tf.constant(0.0,dtype=tf.float64),N-nwidth)
    addH = tf.math.maximum(tf.constant(0.0,dtype=tf.float64),N-nheight)

    new_tl_x = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), new_tl_x - addW/2
    )
    new_tl_y = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), new_tl_y - addH/2
    )
    new_br_x = tf.math.minimum(
        tf.cast(image_shape[1] - 1, dtype=tf.float64),
        new_br_x  + addW/2,
    )
    new_br_y = tf.math.minimum(
        tf.cast(image_shape[0] - 1, dtype=tf.float64),
        new_br_y + addH/2,
    )

    new_tl_x = tf.math.floor(new_tl_x)
    new_tl_y = tf.math.floor(new_tl_y)
    new_br_x = tf.math.floor(new_br_x)
    new_br_y = tf.math.floor(new_br_y)
    #new_w = new_br_x -new_tl_x
    #new_h = 
    return tf.cast(
        tf_reshape_slice([new_tl_x, new_tl_y, new_br_x, new_br_y], shape=2, **kwargs),
        dtype=tf.int32,
    )



@tf.function
def tf_bivariate_normal_pdf(
    mean: tf.Tensor, stddev: tf.Tensor, shape: tf.Tensor, precision=tf.dtypes.float32
) -> tf.Tensor:
    """Produce a heatmap given a Bivariate normal propability density function

    Args:
        mean (tf.Tensor): Mean Tensor(tf.dtype.float*) as [m_x, m_y]
        stddev (tf.Tensor): Standard Deviation Tensor(tf.dtype.float*) as [stdd_x, stdd_y]
        shape (tf.Tensor): Heatmap shape Tensor(tf.dtype.float*) as [width, height]
        precision (tf.dtypes, optional): Precision of the output tensor. Defaults to tf.dtypes.float32.

    Returns:
        tf.Tensor: Heatmap
    """
    # Compute Grid
    X, Y = tf.meshgrid(
        tf.range(
            start=0.0, limit=tf.cast(shape[0], precision), delta=1.0, dtype=precision
        ),
        tf.range(
            start=0.0, limit=tf.cast(shape[1], precision), delta=1.0, dtype=precision
        ),
    )
    mean = tf.round(mean)
    #R = tf.sqrt(((X - mean[0]) ** 2 / (stddev[0])) + ((Y - mean[1]) ** 2 / (stddev[1])))
    R1 = tf.math.square((X - mean[0])/stddev[0]) + tf.math.square((Y - mean[1])/stddev[1])
    #R2 = tf.math.square((X - mean[0])/(4.0*stddev[0])) + tf.math.square((Y - mean[1])/(4.0*stddev[1]))
    #R3 = tf.math.square((X - mean[0])/(16.0*stddev[0])) + tf.math.square((Y - mean[1])/(16.0*stddev[1]))
    #factor = tf.cast(1.0 / (2.0 * m.pi * tf.reduce_prod(stddev)), precision)
    #Z = factor * tf.exp(-0.5 * R)
    #Z1 = 0.75*tf.exp(-0.5*R1)
    #Z2 = 0.16*tf.exp(-0.5*R2)
    #Z3 = 0.09*tf.exp(-0.5*R3)
    #Z = tf.cast(tf.math.floor(255.0*tf.exp(-0.5*R)),dtype=tf.dtypes.uint8)
    #Z = tf.cast(tf.math.floor(255.0*(Z1+Z2+Z3)),dtype=tf.dtypes.uint8)
    Z = tf.exp(-0.5*R1)
    return Z

@tf.function
def tf_hm_distance(
    mean: tf.Tensor, stddev: tf.Tensor, shape: tf.Tensor, precision=tf.dtypes.float32
) -> tf.Tensor:
    """Produce a heatmap given a Bivariate normal propability density function

    Args:
        mean (tf.Tensor): Mean Tensor(tf.dtype.float*) as [m_x, m_y]
        stddev (tf.Tensor): Standard Deviation Tensor(tf.dtype.float*) as [stdd_x, stdd_y]
        shape (tf.Tensor): Heatmap shape Tensor(tf.dtype.float*) as [width, height]
        precision (tf.dtypes, optional): Precision of the output tensor. Defaults to tf.dtypes.float32.

    Returns:
        tf.Tensor: Heatmap
    """
    # Compute Grid
    X, Y = tf.meshgrid(
        tf.range(
            start=0.0, limit=tf.cast(shape[0], precision), delta=1.0, dtype=precision
        ),
        tf.range(
            start=0.0, limit=tf.cast(shape[1], precision), delta=1.0, dtype=precision
        ),
    )
    #mean = tf.floor(mean)
    #R = tf.sqrt(((X - mean[0]) ** 2 / (stddev[0])) + ((Y - mean[1]) ** 2 / (stddev[1])))
    R = tf.sqrt(tf.math.square((X - mean[0])/stddev[0]) + tf.math.square((Y - mean[1])/stddev[1]))
    Rmax = tf.sqrt(tf.constant(2.0,dtype=precision))*tf.constant(64.0,dtype=precision)
    R_d = tf.cast(tf.clip_by_value(255.0*R/Rmax,0.0,255.0),dtype=tf.dtypes.uint8)
    #Z = tf.cast(tf.math.floor(255.0*(Z1+Z2+Z3)),dtype=tf.dtypes.uint8)
    #Z = tf.exp(-0.5*R)
    return R_d


@tf.function
def tf_matrix_argmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply a 2D argmax to a tensor

    Args:
        tensor (tf.Tensor): 3D Tensor with data format HWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension Cx2
    """
    flat_tensor = tf.reshape(tensor, (-1, tf.shape(tensor)[-1]))
    argmax = tf.cast(tf.argmax(flat_tensor, axis=0), tf.int32) # A tensor 1xC
    argmax_x = argmax // tf.shape(tensor)[1]
    argmax_y = argmax % tf.shape(tensor)[1]
    # stack and return 2D coordinates
    return tf.transpose(tf.stack((argmax_x, argmax_y), axis=0), [1, 0])


@tf.function
def tf_batch_matrix_argmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply 2D argmax along a batch

    Args:
        tensor (tf.Tensor): 4D Tensor with data format NHWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxCx2
    """
    return tf.map_fn(
        fn=tf_matrix_argmax, elems=tensor, fn_output_signature=tf.dtypes.int32
    )


@tf.function
def tf_dynamic_matrix_argmax(
    tensor: tf.Tensor, keepdims: bool = True, intermediate_supervision: bool = True
) -> tf.Tensor:
    """Apply 2D argmax for 5D, 4D, 3D, 2D tensors

    This function consider the following dimension cases:
        * `2D tensor` A single joint heatmap.
            Function returns a tensor of `dim=2`.
        * `3D tensor` A multiple joint heatmap.
            Functionelif len(tf.shape(tensor)) == 5:
        # Batch of multiple Joint Heatmaps with Intermediate supervision
        # Considering NSHWC    
        argmax = tf_batch_matrix_argmax(tensor[:, -1, :, :, :])
        return argmax # -> Tensor NxCx2 returns a tensor of `dim=2`.
        * `4D tensor` A multiple joints heatmap with intermediate supervision.
            Function returns a tensor of `dim=2`.
            2D Argmax will only be applied on last stage.
        * `5D tensor` A batch of multiple joints heatmap with intermediate supervision.
            Function returns a tensor of `dim=3`.
            2D Argmax will only be applied on last stage.

    Notes:
        For a batch of heatmap with no intermediate supervision, you need to apply
        a dimension expansion before using this function.
        >>> batch_tensor_no_supervision.shape
        [4, 64, 64, 16]
        >>> tf_dynamic_matrix_argmax(batch_tensor_no_supervision).shape
        [16, 2] # Considered as a single heatmap with intermediate supervision

        >>> expanded_batch = tf.expand_dims(batch_tensor_no_supervision, 1)
        >>> expanded_batch.shape
        [4, 1, 64, 64, 16]
        >>> tf_dynamic_matrix_argmax(batch_tensor_no_supervision).shape
        [4, 16, 2] # Considered as a batch of 4 image


    Args:
        tensor (tf.Tensor): Tensor to apply argmax
        keepdims (bool, optional): Force return tensor to be 3D.
            Defaults to True.
        intermediate_supervision (bool, optional): Modify function behavior if tensor rank is 4.
            Defaults to True.

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxCx2

    Raises:
        ValueError: If the input `tensor` rank not in [2 - 6]
    """
    if len(tf.shape(tensor)) == 2:
        # Single Joint
        argmax = tf_matrix_argmax(tf.expand_dims(tensor, -1))
        return argmax if keepdims else argmax[0, :]
    elif len(tf.shape(tensor)) == 3:
        # Multiple Joint Heatmaps
        argmax = tf_matrix_argmax(tensor)
        return tf.expand_dims(argmax, 0) if keepdims else argmax
    elif len(tf.shape(tensor)) == 4 and intermediate_supervision:
        # Multiple Joint Heatmaps with Intermediate supervision
        # Format SHWC
        argmax = tf_matrix_argmax(tensor[-1, :, :, :])
        return tf.expand_dims(argmax, 0) if keepdims else argmax
    elif len(tf.shape(tensor)) == 4 and not intermediate_supervision:
        # Batch of multiple Joint Heatmaps without Intermediate supervision
        # Considering NHWC   
        argmax = tf_batch_matrix_argmax(tensor)
        return argmax # -> Tensor NxCx2 
    elif len(tf.shape(tensor)) == 5:
        # Batch of multiple Joint Heatmaps with Intermediate supervision
        # Considering NSHWC
        #print(f"{__name__}, Tensor shape: {tensor.shape.as_list()}")    
        argmax = tf_batch_matrix_argmax(tensor[:, -1, :, :, :])
        return argmax # -> Tensor NxCx2
    else:
        raise ValueError(
            f"No argmax operation available for {len(tf.shape(tensor))}D tensor"
        )
