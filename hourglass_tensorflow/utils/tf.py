import math as m
from typing import Tuple,List
import tensorflow as tf
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
    fname = tf.io.read_file(filename)
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
    return tf.image.resize(tensor, size=[size, size], method="nearest") # PREVIOSLY nearest


@tf.function
def tf_rotate_tensor(tensor: tf.Tensor,tshape:tf.Tensor, angle: tf.Tensor,scale: tf.Tensor,center: tf.Tensor) -> tf.Tensor:
    precision = tf.dtypes.float64
    N = tshape[0]
    M = tshape[1]
    _tencenter = tf.convert_to_tensor([N,M],dtype=precision)/2.0
    _tencenter = tf.reshape(_tencenter,shape=[-1,1])
    Ki = tf.range(0,N,1)
    Kj = tf.range(0,M,1)
    # Compute the rotation matrix and translation vector
    center = tf.cast(center,dtype=precision)
    angle = tf.cast(angle,dtype=precision)*np.pi/180.0
    alpha = tf.cast(scale,dtype=precision)*tf.math.cos(angle)
    beta = -1.0*tf.cast(scale,dtype=precision)*tf.math.sin(angle)
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]])
    # Generate pairs of pixel positions for the rotated image
    I,J = tf.meshgrid(Ki,Kj,indexing="ij")
    I = tf.expand_dims(I,axis=2) #Y
    J = tf.expand_dims(J,axis=2) #X
    indexes = tf.concat([I,J],axis=2) # (Y,X)s
    indexes = tf.reshape(indexes,shape=[-1,2]) #Nx2
    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),precision) #2xN
    _center = tf.reshape([center[1],center[0]],[-1,1])
    # Generate indexes from the original image (regarded as the inverse rotation)
    inv_indexes = R@(indexes - _center) + _center #+_center

    inv_indexes_col = tf.clip_by_value(inv_indexes[1],0.00001,1.0*(tf.cast(M,dtype=precision)-1.0))
    inv_indexes_row = tf.clip_by_value(inv_indexes[0],0.00001,1.0*(tf.cast(N,dtype=precision)-1.0))
    
    inv_indexes_col = tf.expand_dims(inv_indexes_col,axis=0)
    inv_indexes_row = tf.expand_dims(inv_indexes_row,axis=0)

    inv_indexes = tf.transpose(tf.concat([inv_indexes_row,inv_indexes_col],axis=0),perm=[1,0])
    inv_indexes = tf.cast(tf.math.floor(inv_indexes),dtype=tf.dtypes.int32)
    pixels_original = tf.gather_nd(tensor,inv_indexes)

    # Fill the whole rotated image with the data sampled from the inverse-rotated image (original)
    indexes = tf.cast(tf.transpose(indexes,perm=[1,0]),tf.dtypes.int32)
    img_rotated = 1.5*tf.zeros_like(tensor) 
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
def tf_rotate_coords(coordinates: tf.Tensor,tshape:tf.Tensor,center:tf.Tensor, visibility:tf.Tensor ,angle: tf.Tensor,scale: tf.Tensor) -> tf.Tensor:

    precision = tf.dtypes.float64
    coords = tf.cast(coordinates,dtype=precision)
    annotated = tf.cast(tf.reduce_sum(coordinates,axis=-1),dtype=precision)
    annotated = tf.cast(tf.where(annotated<0.0,0.0,1.0),dtype=precision)
    
    # Compute the rotation matrix and translation vector
    angle = tf.cast(angle,dtype=precision)*np.pi/180.0
    alpha = tf.cast(scale,dtype=precision)*tf.math.cos(angle)
    beta = -1.0*tf.cast(scale,dtype=precision)*tf.math.sin(angle)
    R = tf.convert_to_tensor([[alpha,beta],[-beta,alpha]]) 

    ymax = tf.cast(tf.reduce_sum(tshape*tf.constant([1,0,0])),dtype=precision)
    xmax = tf.cast(tf.reduce_sum(tshape*tf.constant([0,1,0])),dtype=precision)
    # Compute the rotated coordinates
    _coordinates = tf.cast(tf.transpose(coordinates,perm=[1,0]),dtype=precision)
    _center = tf.cast(tf.reshape(center,shape=(-1,1)),dtype=precision)
    _tencenter = tf.convert_to_tensor([xmax,ymax],dtype=precision)/2.0
    _tencenter = tf.reshape(_tencenter,shape=[-1,1])

    rcoordinates = tf.cast(R@(_coordinates-_center)+_center,precision)#-diff
    
    rcoords_x = rcoordinates[0,:] #tf.clip_by_value(rcoordinates[0,:],0,xmax-1.0)
    vis_x = tf.where(tf.logical_and(rcoords_x<0,rcoords_x>xmax),False,True)
    rcoords_y = rcoordinates[1,:] #tf.clip_by_value(rcoordinates[1,:],0,ymax-1.0)
    vis_y = tf.where(tf.logical_and(rcoords_y<0,rcoords_y>ymax),False,True)
    #vis = tf.reshape(tf.cast(tf.logical_and(tf.logical_and(vis_x,vis_y),tf.cast(visibility,tf.bool)),dtype=precision),(-1,1))
    vis = tf.reshape(tf.cast(visibility,dtype=precision),(-1,1))*annotated
    rcoords = tf.stack([rcoords_x,rcoords_y],axis=0)
    rcoords = tf.cast(tf.transpose(rcoords,perm=[1,0]),dtype=precision)
    rcoords = tf.where(coords<0,coords,rcoords)
    rcoords = tf.concat([rcoords,vis],axis=1) #vis
    
    return tf.cast(rcoords,tf.dtypes.float32)

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
def tf_compute_bbox(coordinates: tf.Tensor,annotated: tf.Tensor, **kwargs) -> tf.Tensor:
    """From a 2D coordinates tensor compute the bounding box

    Args:
        coordinates (tf.Tensor): Joint coordinates 2D tensor

    Returns:
        tf.Tensor: Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    njoints = tf.shape(coordinates)[0]
    #vis = tf.reshape(visibilities,shape=(njoints,1))
    oshape = tf.convert_to_tensor([1,njoints])
    Xs = tf.reshape(tf.cast(coordinates[:, 0],dtype=tf.float32),oshape)#(1,njoints))
    Ys = tf.reshape(tf.cast(coordinates[:, 1],dtype=tf.float32),oshape)#(1,njoints))
    maxx = tf.reduce_max(Xs)
    maxy = tf.reduce_max(Ys)
    #viszeros= tf.zeros(oshape,dtype=tf.float32)#,(1,njoints))
    #visinf = 100000*tf.ones(oshape,dtype=tf.float32)#tf.reshape(tf.constant([100000]*njoints,dtype=tf.float32),(1,njoints))
    #vis = tf.reshape(tf.where(tf.math.logical_and(Xs<0,Ys<0),visinf,viszeros),oshape)#,(1,njoints))
    #viszero = 100000*(1-vis)
    vis = 100000000.0*(1.0-annotated)
    minx = tf.reduce_min(Xs+vis)
    miny = tf.reduce_min(Ys+vis)
    return tf_reshape_slice([minx, miny, maxx, maxy], shape=2, **kwargs)

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
    precision = tf.dtypes.float32
    # Unpack BBox

    top_left = bbox[0]
    top_left_x, top_left_y = tf.cast(top_left[0], dtype=precision), tf.cast(
        top_left[1], dtype=precision
    )
    bottom_right = bbox[1]
    bottom_right_x, bottom_right_y = tf.cast(
        bottom_right[0], dtype=precision
    ), tf.cast(bottom_right[1], dtype=precision)
    # Compute Bbox H/W
    height, width = bottom_right_y - top_left_y, bottom_right_x - top_left_x

    N = tf.maximum(height,width)
    bfactorW = (tf.minimum((N/width),1.05)+0.4*(1.0-(width/N)))*bbox_factor
    bfactorH = (tf.minimum((N/height),1.05)+0.4*(1.0-(height/N)))*bbox_factor

    # Increase BBox Size
    c_tl_x =  top_left_x - width * (bfactorW - 1.0)/2
    c_tl_y = top_left_y - height * (bfactorH - 1.0)/2
    c_br_x = bottom_right_x + width * (bfactorW - 1.0)/2
    c_br_y = bottom_right_y + height * (bfactorH - 1.0)/2
    
    new_tl_x = tf.math.maximum(
        tf.constant(0.0, dtype=precision), c_tl_x
    )
    new_tl_y = tf.math.maximum(
        tf.constant(0.0, dtype=precision), c_tl_y
    )
    new_br_x = tf.math.minimum(
        tf.cast(image_shape[1] - 1, dtype=precision),
        c_br_x
    )
    new_br_y = tf.math.minimum(
        tf.cast(image_shape[0] - 1, dtype=precision),
        c_br_y
    )

    ptlx = new_tl_x - c_tl_x
    ptly = new_tl_y - c_tl_y 
    pbrx = c_br_x - new_br_x 
    pbry = c_br_y -new_br_y    
    padding_tensor = tf.convert_to_tensor([
        [ptly, pbry],
        [ptlx, pbrx],
        [0, 0],
    ],dtype=tf.int32)

    new_tl_x = tf.math.floor(new_tl_x)
    new_tl_y = tf.math.floor(new_tl_y)
    new_br_x = tf.math.floor(new_br_x)
    new_br_y = tf.math.floor(new_br_y)

    return tf.cast(
        tf_reshape_slice([new_tl_x, new_tl_y, new_br_x, new_br_y], shape=2, **kwargs),
        dtype=tf.int32,
    ), padding_tensor


@tf.function
def tf_3Uint8_to_float32(
    tensor: tf.Tensor
) -> tf.Tensor:
    _tensor = tf.cast(tensor,dtype=tf.float32)
    out_tensor = tf.zeros(tf.shape(tensor)[0:2],dtype=tf.float32)
    out_tensor += _tensor[:,:,2]*(256**2)+_tensor[:,:,1]*256.0+_tensor[:,:,0]
    return out_tensor

    

@tf.function
def tf_bivariate_normal_pdf(
    mean: tf.Tensor,vis: tf.Tensor , stddev: tf.Tensor, shape: tf.Tensor, precision=tf.dtypes.float32
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
    if vis==1:
        X, Y = tf.meshgrid(
            tf.range(
                start=0.0, limit=tf.cast(shape[0], precision), delta=1.0, dtype=precision
            ),
            tf.range(
                start=0.0, limit=tf.cast(shape[1], precision), delta=1.0, dtype=precision
            ),
        )
        
        #mean = tf.clip_by_value(tf.round(mean),0,63)
        mean = tf.clip_by_value(mean,0,63)
        
        #R = tf.sqrt(tf.math.square((X - mean[0])) + tf.math.square((Y - mean[1])))/stddev[0]
        #R = tf.sqrt(((X - mean[0]) ** 2 / (stddev[0])) + ((Y - mean[1]) ** 2 / (stddev[1])))
        #factor = tf.cast(1.0 / (2.0 * m.pi * tf.reduce_prod(stddev)), precision)

        #R1 = tf.math.square((X - mean[0])/stddev[0]) + tf.math.square((Y - mean[1])/stddev[1])
        R1 = tf.math.square((X - mean[0])/stddev) + tf.math.square((Y - mean[1])/stddev)
        Z = tf.exp(-0.5*R1)#-0.000001 #+ 0.00001#- 0.0001
    else:
        Z = 0.000001*tf.ones(tf.cast(shape, dtype=tf.dtypes.int32), dtype=precision)#-0.001*tf.ones(tf.cast(shape, dtype=tf.dtypes.int32), dtype=precision)
    #R2 = tf.math.square((X - mean[0])/(4.0*stddev[0])) + tf.math.square((Y - mean[1])/(4.0*stddev[1]))
    #R3 = tf.math.square((X - mean[0])/(16.0*stddev[0])) + tf.math.square((Y - mean[1])/(16.0*stddev[1]))
    #factor = tf.cast(1.0 / (2.0 * m.pi * tf.reduce_prod(stddev)), precision)
    #Z = factor * tf.exp(-0.5 * R)
    #Z1 = 0.75*tf.exp(-0.5*R1)
    #Z2 = 0.16*tf.exp(-0.5*R2)
    #Z3 = 0.09*tf.exp(-0.5*R3)
    #Z = tf.cast(tf.math.floor(255.0*tf.exp(-0.5*R)),dtype=tf.dtypes.uint8)
    #Z = tf.cast(tf.math.floor(255.0*(Z1+Z2+Z3)),dtype=tf.dtypes.uint8)
    return Z

@tf.function
def tf_generate_segment(idx0, idx1,joints):
    pnt0 = tf.reshape(joints[idx0,0:2],(1,2))
    pnt1 = tf.reshape(joints[idx1,0:2],(1,2))
    vis0 = joints[idx0,2]
    vis1 = joints[idx1,2]
    vis = tf.convert_to_tensor([vis0,vis1],dtype=tf.float32)
    vis = tf.reshape(vis,(1,2))
    segment = tf.concat([pnt0,pnt1,vis],axis=0)
    return tf.cast(segment,tf.float32)

@tf.function
def tf_bivariate_segment_normal_pdf(
    points: tf.Tensor,vis: tf.Tensor , stddev: tf.Tensor, shape: tf.Tensor, precision=tf.dtypes.float32
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
    #means = tf.round(points)
    means = tf.clip_by_value(points,0,63)
    means = tf.cast(means,dtype=tf.float32)
    signs = tf.sign(means)
    signs = tf.sign(tf.reduce_sum(signs,axis=1))

    if vis[0]==1 or vis[1]==1 and signs[0]==1 and signs[1]==1:
        X, Y = tf.meshgrid(
            tf.range(
                start=0.0, limit=tf.cast(shape[0], precision), delta=1.0, dtype=precision
            ),
            tf.range(
                start=0.0, limit=tf.cast(shape[1], precision), delta=1.0, dtype=precision
            ),
        )
        D1 = tf.math.sqrt(tf.math.square(X - means[0][0]) + tf.math.square(Y - means[0][1])) 
        D2 = tf.math.sqrt(tf.math.square(X - means[1][0]) + tf.math.square(Y - means[1][1])) 
        DS = tf.math.sqrt(tf.math.square(means[1][0] - means[0][0]) + tf.math.square(means[1][1] - means[0][1]))
        R = tf.math.square((D1+D2-DS)/2.0)/(tf.square(stddev))
        Z = tf.exp(-0.5*R)
    elif vis[0]==1 and vis[1]==0 and signs[0]==1 and signs[1]==-1 and False:
        Z = tf_bivariate_normal_pdf(mean=means[0],vis=vis[0] , stddev=stddev, shape=shape, precision=precision)
    elif vis[1]==1 and vis[0]==0 and signs[1]==1 and signs[0]==-1 and False:
        Z = tf_bivariate_normal_pdf(mean=means[1],vis=vis[1] , stddev=stddev, shape=shape, precision=precision)
    else:
        Z = 0.00001*tf.ones(tf.cast(shape, dtype=tf.dtypes.int32), dtype=precision)
    return Z


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
    argmax_x = argmax % tf.shape(tensor)[1] # //
    argmax_y = argmax // tf.shape(tensor)[1] # % 
    # stack and return 2D coordinates
    return tf.transpose(tf.stack((argmax_x, argmax_y), axis=0), [1, 0])


@tf.function
def tf_matrix_softargmax_loss(tensor: tf.Tensor) -> tf.Tensor:
    """Apply a 2D argmax to a tensor

    Args:
        tensor (tf.Tensor): 3D Tensor with data format HWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension Cx2
    """
    #_tensor = tf.nn.relu(tensor)
    _tens_min = tf.reduce_min(tensor,axis=[0,1],keepdims=True)
    _tens_max = tf.reduce_max(tensor,axis=[0,1],keepdims=True)
    _tensor = (tensor-_tens_min)/(_tens_max-_tens_min+0.0001)
    #thresh_tensor = tf.where(_tensor > 0.3, _tensor, 0.3*tf.ones_like(tensor))
    _flat_tensor = tf.reshape(100.0*_tensor, (-1, tf.shape(tensor)[-1]))
    flat_shape = tf.shape(_flat_tensor)
    val = 64*64-32
    _zero_correction = tf.reshape(tf.convert_to_tensor([1.5]*32+[0.0]*val),shape=(flat_shape[0],1))
    _flat_tensor = _flat_tensor + tf.cast(_zero_correction,dtype=tf.float32)
    # Apply softmax to normalize heatmaps
    flat_tensor = tf.nn.softmax(_flat_tensor, axis=0) #HWxC 

    # Create coordinate grids
    x_grid = tf.range(tf.shape(tensor)[0], dtype=tf.float32)
    y_grid = tf.range(tf.shape(tensor)[1], dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x_grid, y_grid)

    # Flatten coordinate grids
    x_grid = tf.reshape(x_grid, shape=(-1,1)) #HWx1
    y_grid = tf.reshape(y_grid, shape=(-1,1)) #HWx1 

    # Compute expected (x, y) coordinates using softmax weights
    x = tf.reduce_sum(x_grid * flat_tensor, axis=0) #C,
    y = tf.reduce_sum(y_grid * flat_tensor, axis=0) #C,  
    # stack and return 2D coordinates
    return tf.transpose(tf.stack((x,y), axis=0), [1, 0])

@tf.function
def tf_multistage_matrix_softargmax_loss(tensor: tf.Tensor) -> tf.Tensor:
    """Apply 2D argmax along multiple stages

    Args:
        tensor (tf.Tensor): 4D Tensor with data format SHWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension SxCx2
    """
    return tf.map_fn(
        fn=tf_matrix_softargmax_loss, elems=tensor, fn_output_signature=tf.dtypes.float32
    )

@tf.function
def tf_batch_multistage_matrix_softargmax_loss(tensor: tf.Tensor) -> tf.Tensor:
    """Apply 2D argmax along a batch

    Args:
        tensor (tf.Tensor): 4D Tensor with data format SHWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxSxCx2
    """
    return tf.map_fn(
        fn=tf_multistage_matrix_softargmax_loss, elems=tensor, fn_output_signature=tf.dtypes.float32
    )

@tf.function
def tf_matrix_softargmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply a 2D argmax to a tensor

    Args:
        tensor (tf.Tensor): 3D Tensor with data format HWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension Cx2
    """
    _tensor = tf.nn.relu(tensor)
    _tensor = tensor/(tf.reduce_max(_tensor,axis=[0,1],keepdims=True)+0.0001)
    
    #thresh_tensor = tf.where(_tensor > 0.3, _tensor, 0.3*tf.ones_like(tensor))
    _flat_tensor = tf.reshape(100.0*_tensor, (-1, tf.shape(tensor)[-1]))
    # Apply softmax to normalize heatmaps
    flat_tensor = tf.nn.softmax(_flat_tensor, axis=0) #HWxC 

    # Create coordinate grids
    x_grid = tf.range(tf.shape(tensor)[0], dtype=tf.float32)
    y_grid = tf.range(tf.shape(tensor)[1], dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x_grid, y_grid)

    # Flatten coordinate grids
    x_grid = tf.reshape(x_grid, shape=(-1,1)) #HWx1
    y_grid = tf.reshape(y_grid, shape=(-1,1)) #HWx1 

    # Compute expected (x, y) coordinates using softmax weights
    x = tf.reduce_sum(x_grid * flat_tensor, axis=0) #C,
    y = tf.reduce_sum(y_grid * flat_tensor, axis=0) #C,  
    # stack and return 2D coordinates
    return tf.transpose(tf.stack((x,y), axis=0), [1, 0])

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
def tf_batch_matrix_softargmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply 2D argmax along a batch

    Args:
        tensor (tf.Tensor): 4D Tensor with data format NHWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxCx2
    """
    return tf.map_fn(
        fn=tf_matrix_softargmax, elems=tensor, fn_output_signature=tf.dtypes.float32
    )


@tf.function
def tf_normalize_tensor(tensor:tf.Tensor,thresh_val: float) -> tf.Tensor:
    """
    Apply normalization disregarding the zero entries
    """
    mask = tf.where(tensor<=thresh_val,0.0,1.0)
    numpx = tf.reduce_sum(mask)
    mean = tf.reduce_sum(tensor*mask)/numpx
    stddev = tf.sqrt(tf.reduce_sum(tf.square(tensor*mask-mean))/numpx+0.0000001)
    _tensor = (tensor-mean)/stddev
    return tf.clip_by_value(1.5*(_tensor+3.5)+1.5,0.0,800.0)*mask

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
