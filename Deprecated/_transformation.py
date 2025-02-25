import tensorflow as tf
from typing import List,Tuple
import cv2
import numpy as np

from hourglass_tensorflow.utils.tf import tf_stack
from hourglass_tensorflow.utils.tf import tf_load_image
from hourglass_tensorflow.utils.tf import tf_expand_bbox
from hourglass_tensorflow.utils.tf import tf_compute_bbox,tf_compute_bbox_bc
from hourglass_tensorflow.utils.tf import tf_reshape_slice
from hourglass_tensorflow.utils.tf import tf_resize_tensor
from hourglass_tensorflow.utils.tf import tf_bivariate_normal_pdf
from hourglass_tensorflow.utils.tf import tf_generate_padding_tensor
from hourglass_tensorflow.utils.tf import tf_compute_padding_from_bbox
from hourglass_tensorflow.utils.tf import tf_rotate_tensor,tf_rotate_coords,tf_rotate_norm_coords, tf_hm_distance
from hourglass_tensorflow.utils.tf import tf_squarify_image_scale,tf_squarify_coordinates_scale,tf_expand_bbox_squared
from hourglass_tensorflow.utils.tf import tf_3Uint8_to_float32,tf_generate_segment, tf_bivariate_segment_normal_pdf


@tf.function
def tf_train_map_build_slice_RGB(filename: tf.Tensor, coordinates: tf.Tensor) -> tf.Tensor:
    """First step loader for tf.data.Dataset mapper

    This mapper is used on Training phase only to load images and shape coordinates.

    Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        filename (tf.Tensor): string tensor containing the image path to read
        coordinates (tf.Tensor): _description_

    Returns:
        tf.Tensor: _description_
    """
    # Load Image
    #print("RESHAPED ",tf.squeeze(filename))
    _fname = tf.squeeze(filename)
    image = tf_load_image(_fname)
    img_shape = tf.shape(image)
    # Shape coordinates
    joints = tf_reshape_slice(coordinates, shape=3)
    # Extract coordinates and visibility from joints
    coordinates = joints[:, :2]
    visibility = joints[:, 2]
    return (image, coordinates, visibility,img_shape)

@tf.function
def tf_train_map_build_slice_RGBD(filename_rgb: tf.Tensor, filename_depth: tf.Tensor, coordinates: tf.Tensor) -> tf.Tensor:
    """First step loader for tf.data.Dataset mapper

    This mapper is used on Training phase only to load images and shape coordinates.

    Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        filename (tf.Tensor): string tensor containing the image path to read
        coordinates (tf.Tensor): _description_

    Returns:
        tf.Tensor: _description_
    """
    # Load Image
    #print("RESHAPED ",tf.squeeze(filename))
    _fnamergb = tf.squeeze(filename_rgb)
    _fnamergbD = tf.squeeze(filename_depth)

    imagergb = tf_load_image(_fnamergb)
    imagedepth = tf_load_image(_fnamergbD)

    #depthmap = tf.expand_dims(tf.clip_by_value((tf_3Uint8_to_float32(imagedepth)-100.0)*(255.0/2500.0),0.0,255.0),axis=2)
    depthmap = tf.expand_dims(tf_3Uint8_to_float32(imagedepth),axis=2)
    meandepth = tf.reduce_mean(depthmap,axis=[0,1,2])
    stddevdepth =  tf.sqrt(tf.reduce_mean(tf.square(depthmap-meandepth),axis=[0,1,2]))
    depthmap = 128.0*(depthmap-meandepth)/stddevdepth
    RGBD_image = tf.concat([tf.cast(imagergb,dtype=tf.float32),depthmap],axis=2)
    img_shape = tf.shape(RGBD_image)
    # Shape coordinates
    joints = tf_reshape_slice(coordinates, shape=3)
    # Extract coordinates and visibility from joints
    coordinates = joints[:, :2]
    visibility = joints[:, 2]
    return (RGBD_image, coordinates, visibility,img_shape)


@tf.function
def tf_train_map_affine_augmentation_RGB(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64,
    njoints: int = 16,
    hip: Tuple[int,int] = [3,2]
    )-> tf.Tensor:
    """
    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
    """

    # Generate one image for each rotation angle in rotation_angles (Input: image)
    # Generate one coordinate Tensor for each rotation (Input: coordinates)
    affines = tf.constant([[-20,1.0],
                           [-18,1.0],
                           [-16,1.0],
                            [-12.0,1.0],
                            [-10.0,1.0],
                            [-8.0,1.0],
                           [0,1.0],
                           [0,1.0],
                           [0,1.0],
                           [8.0,1.0],
                            [10.0,1.0],
                            [12.0,1.0],
                           [16,1.0],
                           [18,1.0],
                           [20,1.0]],dtype=tf.float32)
    
    affines = tf.constant([[-20,1.0],
                            [-20,1.0],
                            [-16,1.0],
                            [-16,1.0],
                            [-12.0,1.0],
                            [-12.0,1.0],
                            [-8.0,1.0],
                            [-8.0,1.0],
                            [0,1.0],
                            [0,1.0],
                            [8.0,1.0],
                            [8.0,1.0],
                            [12.0,1.0],
                            [12.0,1.0],
                            [16,1.0],
                            [16,1.0],
                            [20,1.0],
                            [20,1.0]],dtype=tf.float32)


    # Hip center
    
    center = tf.reduce_mean(tf.cast(tf_compute_bbox(coordinates),tf.float32),axis=0)
    #center = tf.reduce_mean(coordinates,tf.float32,axis=1)
    #center = 0.5*(tf.cast(coordinates[hip[0]]+coordinates[hip[1]],dtype=tf.float32))
    center = tf.reshape(center,shape=(2,))
    image = tf.cast(image,dtype=tf.float32)
    _images = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_tensor(image,
                                            img_shape,
                                            affine[0],
                                            affine[1],
                                            center,
            )
        ),
        elems=affines,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates_map = tf.map_fn(
        fn=(
            #lambda affine: tf_rotate_norm_coords(coordinates,
            lambda affine: tf_rotate_coords(coordinates,
                                            img_shape,
                                            center,
                                            visibility,
                                            affine[0],
                                            affine[1]
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        #dtype=tf.dtypes.float64,
        parallel_iterations=10,
    )
    _visibilities  = _coordinates_map[:,:,2]
    _coordinates = _coordinates_map[:,:,0:2]
    
    
    #_visibilities = tf.stack([visibility]*18, axis=0)

    _bboxf = tf.constant([1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14],
                         dtype=tf.float32)
    
    _zipped = tf.map_fn(
        fn=(
            lambda imgncoords: tf_train_map_squarify(imgncoords[0],
                                                     imgncoords[1],
                                                     imgncoords[2],
                                                     True,
                                                     imgncoords[3])
        ),
        elems=(tf.cast(_images,dtype=tf.float32),
               tf.cast(_coordinates,dtype=tf.float32),
               tf.cast(_visibilities,dtype=tf.float32),
               _bboxf),
        parallel_iterations=10,
    )

    """
    _images: a Tensor (R,H,W,3) of several images.
    _coordinates: a Tensor (R,C,2) of coordinates for several rotations
    _visibilities: a Tensor (R,C), just copy the visibility values
    """
    #print("ZIPPED",_zipped[0].shape,type(_zipped),type(_zipped[0]))

    _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.uint8),[18,input_size,input_size,3])
    _coords = tf.reshape(_zipped[1],[18,njoints,2])
    _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[18,njoints])
    
    return (_images,_coords,_visibilities)

@tf.function
def tf_train_map_affine_augmentation_RGBD(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64,
    njoints: int = 16,
    hip: Tuple[int,int] = [3,2]
    )-> tf.Tensor:
    """
    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
    """

    # Generate one image for each rotation angle in rotation_angles (Input: image)
    # Generate one coordinate Tensor for each rotation (Input: coordinates)
    affines = tf.constant([[-12.0,1.0],
                            [-10.0,1.0],
                            [-8.0,1.0],
                            [0,1.0],
                            [0,1.0],
                            [0,1.0],
                            [8.0,1.0],
                            [10.0,1.0],
                            [12.0,1.0]],dtype=tf.float32)
    
    affines = tf.constant([[-20,1.0],
                            [-20,1.0],
                            [-16,1.0],
                            [-16,1.0],
                            [-12.0,1.0],
                            [-12.0,1.0],
                            [-8.0,1.0],
                            [-8.0,1.0],
                            [0,1.0],
                            [0,1.0],
                            [8.0,1.0],
                            [8.0,1.0],
                            [12.0,1.0],
                            [12.0,1.0],
                            [16,1.0],
                            [16,1.0],
                            [20,1.0],
                            [20,1.0]],dtype=tf.float32)

    # Hip center
    #center = tf.floor(0.5*(tf.cast(coordinates[hip[0]]+coordinates[hip[1]],dtype=tf.float32)))
    center = tf.reduce_mean(tf.cast(tf_compute_bbox(coordinates),tf.float32),axis=0)
    image = tf.cast(image,dtype=tf.float32)
    _images = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_tensor(image,
                                            img_shape,
                                            affine[0],
                                            affine[1],
                                            center,
            )
        ),
        elems=affines,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates_map = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_coords(coordinates,
                                            img_shape,
                                            center,
                                            visibility,
                                            affine[0],
                                            affine[1],
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        parallel_iterations=10,
    )

    _visibilities  = _coordinates_map[:,:,2]
    _coordinates = _coordinates_map[:,:,0:2]
    
    
    #_visibilities = tf.stack([visibility]*18, axis=0)
    #_visibilities = tf.stack([visibility]*9, axis=0)

    _bboxf = tf.constant([1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14,1.24,1.14],
                         dtype=tf.float32)
    

    #_bboxf = tf.constant([1.16,1.18,1.16,1.22,1.18,1.16,1.16,1.18,1.18],
    #                     dtype=tf.float32)
    
    _zipped = tf.map_fn(
        fn=(
            lambda imgncoords: tf_train_map_squarify(imgncoords[0],
                                                     imgncoords[1],
                                                     imgncoords[2],
                                                     True,
                                                     imgncoords[3])
        ),
        elems=(tf.cast(_images,dtype=tf.float32),
               tf.cast(_coordinates,dtype=tf.float32),
               tf.cast(_visibilities,dtype=tf.float32),
               _bboxf),
        parallel_iterations=10,
    )

    """
    _images: a Tensor (R,H,W,3) of several images.
    _coordinates: a Tensor (R,C,2) of coordinates for several rotations
    _visibilities: a Tensor (R,C), just copy the visibility values
    """
    #print("ZIPPED",_zipped[0].shape,type(_zipped),type(_zipped[0]))

    _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[18,input_size,input_size,4])
    _coords = tf.reshape(_zipped[1],[18,njoints,2])
    _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[18,njoints])
    return (_images,_coords,_visibilities)


@tf.function
def tf_train_map_affine_woaugment_RGB(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64,
    njoints: int = 16,
    hip: Tuple[int,int] = [3,2],
    task_mode: str = "train"
    )-> tf.Tensor:
    """
    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
    """

    # Generate one image for each rotation angle in rotation_angles (Input: image)
    # Generate one coordinate Tensor for each rotation (Input: coordinates)
    #angles = tf.constant([-30,-15,0,15,30])
    #affines = tf.constant([[-15.0,1.0],[0.0,1.0],[15.0,1.0]],dtype=tf.float32)
    affines = tf.constant([[0.0,1.0]],dtype=tf.float32)

    # Hip center
    #center = tf.floor(0.5*(tf.cast(coordinates[hip[0]]+coordinates[hip[1]],dtype=tf.float32)))
    center = tf.reduce_mean(tf.cast(tf_compute_bbox(coordinates),tf.float32),axis=0)

    image = tf.cast(image,dtype=tf.float32)
    _images = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_tensor(image,
                                            img_shape,
                                            affine[0],
                                            affine[1],
                                            center,
                                            #input_size=input_size,
            )
        ),
        elems=affines,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates_map = tf.map_fn(
        fn=(
            #lambda affine: tf_rotate_norm_coords(coordinates,
            lambda affine: tf_rotate_coords(coordinates,
                                            img_shape,
                                            center,
                                            visibility,
                                            affine[0],
                                            affine[1],
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        #dtype=tf.dtypes.float64,
        parallel_iterations=10,
    )

    _visibilities  = _coordinates_map[:,:,2]
    _coordinates = _coordinates_map[:,:,0:2]
    #_visibilities = tf.stack([visibility]*1, axis=0)

    #_bboxf = tf.constant([1.095,1.08,1.08,1.07,1.05,1.09,1.03,1.08,1.13,1.095,1.08,1.08,1.07,1.05,1.09],
    #                     dtype=tf.float32)
    #_bboxf = tf.constant([1.16,1.16,1.16],dtype=tf.float32) 
    _bboxf = tf.constant([1.20],dtype=tf.float32) 
    _zipped = tf.map_fn(
        fn=(
            lambda imgncoords: tf_train_map_squarify(imgncoords[0],
                                                     imgncoords[1],
                                                     imgncoords[2],
                                                     True,
                                                     imgncoords[3])
        ),
        elems=(tf.cast(_images,dtype=tf.float32),
               tf.cast(_coordinates,dtype=tf.float32),
               tf.cast(_visibilities,dtype=tf.float32),
               _bboxf),
        parallel_iterations=10,
    )

    """
    _images: a Tensor (R,H,W,3) of several images.
    _coordinates: a Tensor (R,C,2) of coordinates for several rotations
    _visibilities: a Tensor (R,C), just copy the visibility values
    """
    _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.uint8),[1,input_size,input_size,3])
    _coords = tf.reshape(_zipped[1],[1,njoints,2])
    _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[1,njoints])
    if task_mode=="train":
        return (_images,_coords,_visibilities)
    elif task_mode=="test":
        return (_images,coordinates,_visibilities)

@tf.function
def tf_train_map_affine_woaugment_RGBD(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64,
    njoints: int = 16,
    hip: Tuple[int,int] = [3,2],
    task_mode: str = "train"
    )-> tf.Tensor:
    """
    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
    """

    # Generate one image for each rotation angle in rotation_angles (Input: image)
    # Generate one coordinate Tensor for each rotation (Input: coordinates)
    if task_mode=="train":
        affines = tf.constant([[0.0,1.0],
                            [0.0,1.0],
                            [0.0,1.0]],dtype=tf.float32)
    elif task_mode=="test":
        affines = tf.constant([[0.0,1.0]],dtype=tf.float32)

   # Hip center
    #center = tf.floor(0.5*(tf.cast(coordinates[hip[0]]+coordinates[hip[1]],dtype=tf.float32)))
    center = tf.reduce_mean(tf.cast(tf_compute_bbox(coordinates),tf.float32),axis=0)

    image = tf.cast(image,dtype=tf.float32)
    _images = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_tensor(image,
                                            img_shape,
                                            affine[0],
                                            affine[1],
                                            center,
                                            #input_size=input_size,
            )
        ),
        elems=affines,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates_map = tf.map_fn(
        fn=(
            #lambda affine: tf_rotate_norm_coords(coordinates,
            lambda affine: tf_rotate_coords(coordinates,
                                            img_shape,
                                            center,
                                            visibility,
                                            affine[0],
                                            affine[1],
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        parallel_iterations=10,
    )

    _visibilities  = _coordinates_map[:,:,2]
    _coordinates = _coordinates_map[:,:,0:2]

    if task_mode=="train":
        _bboxf = tf.constant([1.12,1.16,1.20],dtype=tf.float32) 
    elif task_mode=="test":
        _bboxf = tf.constant([1.18],dtype=tf.float32)
    
    _zipped = tf.map_fn(
        fn=(
            lambda imgncoords: tf_train_map_squarify(imgncoords[0],
                                                     imgncoords[1],
                                                     imgncoords[2],
                                                     True,
                                                     imgncoords[3])
        ),
        elems=(tf.cast(_images,dtype=tf.float32),
               tf.cast(_coordinates,dtype=tf.float32),
               tf.cast(_visibilities,dtype=tf.float32),
               _bboxf),
        parallel_iterations=10,
    )

    """
    _images: a Tensor (R,H,W,3) of several images.
    _coordinates: a Tensor (R,C,2) of coordinates for several rotations
    _visibilities: a Tensor (R,C), just copy the visibility values
    """
    if task_mode == "train":
        _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[3,256,256,4])
        _coords = tf.reshape(_zipped[1],[3,njoints,2])
        _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[3,njoints])
    elif task_mode == "test":
        _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[1,256,256,4])
        _coords = tf.reshape(_zipped[1],[1,njoints,2])
        _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[1,njoints])
        _bboxes = tf.reshape(tf.cast(_zipped[3],dtype=tf.float32),[1,3,2])
    
    if task_mode=="train":
        return (_images,_coords,_visibilities)
    elif task_mode=="test":
        return (_images,coordinates,_bboxes,_visibilities)

@tf.function
def tf_test_map_affine_woaugment_RGBD(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64,
    njoints: int = 16,
    hip: Tuple[int,int] = [3,2],
    task_mode: str = "train"
    )-> tf.Tensor:
    """
    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
    """

    # Generate one image for each rotation angle in rotation_angles (Input: image)
    # Generate one coordinate Tensor for each rotation (Input: coordinates)
    """
    if task_mode=="train":
        affines = tf.constant([[0.0,1.0],
                            [0.0,1.0],
                            [0.0,1.0]],dtype=tf.float32)
    elif task_mode=="test":
        affines = tf.constant([[0.0,1.0]],dtype=tf.float32)
    """
    affines = tf.constant([[0.0,1.0]],dtype=tf.float32)

   # Hip center
    #center = tf.floor(0.5*(tf.cast(coordinates[hip[0]]+coordinates[hip[1]],dtype=tf.float32)))
    center = tf.reduce_mean(tf.cast(tf_compute_bbox(coordinates),tf.float32),axis=0)

    image = tf.cast(image,dtype=tf.float32)
    _images = tf.map_fn(
        fn=(
            lambda affine: tf_rotate_tensor(image,
                                            img_shape,
                                            affine[0],
                                            affine[1],
                                            center,
                                            #input_size=input_size,
            )
        ),
        elems=affines,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates_map = tf.map_fn(
        fn=(
            #lambda affine: tf_rotate_norm_coords(coordinates,
            lambda affine: tf_rotate_coords(coordinates,
                                            img_shape,
                                            center,
                                            visibility,
                                            affine[0],
                                            affine[1],
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        parallel_iterations=10,
    )

    _visibilities  = _coordinates_map[:,:,2]
    _coordinates = _coordinates_map[:,:,0:2]
    """
    if task_mode=="train":
        _bboxf = tf.constant([1.12,1.16,1.20],dtype=tf.float32) 
    elif task_mode=="test":
        _bboxf = tf.constant([1.18],dtype=tf.float32)
    """
    _bboxf = tf.constant([1.18],dtype=tf.float32)
    
    _zipped = tf.map_fn(
        fn=(
            lambda imgncoords: tf_train_map_squarify(imgncoords[0],
                                                     imgncoords[1],
                                                     imgncoords[2],
                                                     True,
                                                     imgncoords[3])
        ),
        elems=(tf.cast(_images,dtype=tf.float32),
               tf.cast(_coordinates,dtype=tf.float32),
               tf.cast(_visibilities,dtype=tf.float32),
               _bboxf),
        parallel_iterations=10,
    )

    """
    _images: a Tensor (R,H,W,3) of several images.
    _coordinates: a Tensor (R,C,2) of coordinates for several rotations
    _visibilities: a Tensor (R,C), just copy the visibility values
    
    if task_mode == "train":
        _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[3,256,256,4])
        _coords = tf.reshape(_zipped[1],[3,njoints,2])
        _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[3,njoints])
    elif task_mode == "test":
        _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[1,256,256,4])
        _coords = tf.reshape(_zipped[1],[1,njoints,2])
        _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[1,njoints])
        _bboxes = tf.reshape(tf.cast(_zipped[3],dtype=tf.float32),[1,3,2])
    
    if task_mode=="train":
        return (_images,_coords,_visibilities)
    elif task_mode=="test":
        return (_images,coordinates,_bboxes,_visibilities)
    """
    _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.float32),[1,256,256,4])
    _coords = tf.reshape(_zipped[1],[1,njoints,2])
    _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[1,njoints])
    _bboxes = tf.reshape(tf.cast(_zipped[3],dtype=tf.float32),[1,3,2])  
    return (_images,tf.expand_dims(coordinates,axis=0),_bboxes,_visibilities)
    

@tf.function
def tf_train_map_squarify(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    bbox_enabled=False,
    bbox_factor=1.0,
) -> tf.Tensor:
    """Second step tf.data.Dataset mapper to make squared input images

    This mapper is used on Training phase only to make a squared image.
    It would not suit Preditction phase since you need to have prior
    knowledge of the person position

    Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
        bbox_enabled (bool, optional): Crop image to fit bbox . Defaults to False
        bbox_factor (float, optional): Expanding factor for bbox. Defaults to 1.0

    Returns:
        tf.Tensor: _description_
    """
    if bbox_enabled:
        # Compute Bounding Box
        #bbox = tf_expand_bbox_squared(
        bbox,add_padding = tf_expand_bbox(
            #tf_compute_bbox_bc(coordinates,tf.shape(image)),
            tf_compute_bbox(coordinates),
            tf.shape(image),
            bbox_factor=bbox_factor,
        )
    else:
        # Simulate a Bbox being the whole image
        shape = tf.shape(image)
        bbox = tf.cast([[0, 0], [shape[1] - 1, shape[0] - 1]])

    #image = tf.pad(
    #    image[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0], :],
    #    paddings=add_padding,
    #)
    #image = image[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0], :]
    nshape = tf.shape(image)
    
    #coordinates = coordinates - tf.cast(bbox[0],dtype=tf.float32)+ tf.convert_to_tensor([add_padding[1,0],add_padding[0,0]],dtype=tf.float32)
    
    # Get Padding
    # Once the bbox is computed we compute
    # how much V/H padding should be applied
    # Padding is necessary to conserve proportions
    # when resizing
    #print(bbox)
    nbbox = tf.cast([[0, 0], [nshape[1] - 1, nshape[0] - 1]],tf.int32)
    padding = tf_compute_padding_from_bbox(bbox) #nbbox
    # Generate Squared Image with Padding
    
    image = tf.pad(image[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0], :],
        paddings=tf_generate_padding_tensor(padding),
    )
    nshape = tf.shape(image)[0:2]
     #tf.convert_to_tensor([bbox[1, 0]-bbox[0, 0],bbox[1, 1]-bbox[0, 1]])
    scale = 256.0/(tf.cast(nshape,dtype=tf.dtypes.float64))
    # Recompute coordinates
    # Given the padding and eventual bounding box
    # we need to recompute the coordinates from
    # a new origin
    
    #coordinates = coordinates - (bbox[0] - padding)
    #bboxf64 = 1.0#/tf.cast(bbox_factor,dtype=tf.dtypes.float64)
    #coord_shift = padding + tf.convert_to_tensor([add_padding[1,0],add_padding[0,0]],dtype=tf.int32)
    
    coordinates = tf.cast(coordinates,dtype=tf.dtypes.float64) - tf.cast(bbox[0] - padding,dtype=tf.dtypes.float64)
    #coordinates = tf.cast(coordinates,dtype=tf.dtypes.float64) + tf.cast(padding,dtype=tf.dtypes.float64)
    _center = 0.5*tf.cast(nshape,dtype=tf.dtypes.float64)
    #= tf.cast(coordinates,dtype=tf.dtypes.float64) - tf.cast(bbox[0] - padding,dtype=tf.dtypes.float64)
    #_center = 0.5*(coordinates[2]+coordinates[3])
    coordinates = scale*(coordinates)#((coordinates-_center)*bboxf64)+_center)
    
    return (
        tf_resize_tensor(image,256),
        tf.cast(coordinates,dtype=tf.dtypes.float32),
        visibility,
        tf.cast(tf.concat([bbox,tf.reshape(padding,(1,-1))],axis=0),tf.float32)
        #visibility,
    )

@tf.function
def tf_train_map_resize_data(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    bboxes: tf.Tensor,
    input_size: int = 256,
    task_mode: str = "train",
) -> tf.Tensor:
    """Third step tf.data.Dataset mapper to reshape image
    and compute relative coordinates.

    This mapper is used on Training phase only.
    It would suit not Preditction phase since you need to have prior
    knowledge of the person position

    Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
        input_size (int, optional): Desired size for the input image. Defaults to 256

    Returns:
        tf.Tensor: _description_
    """
    # Reshape Image
    shape = tf.cast(tf.shape(image), dtype=tf.dtypes.float64)
    image = tf.cast(image,dtype=tf.dtypes.float32)
    
    # We compute the Height and Width reduction factors
    #h_factor = shape[0] / tf.cast(input_size, tf.dtypes.float64)
    #w_factor = shape[1] / tf.cast(input_size, tf.dtypes.float64)
    # We can recompute relative Coordinates between 0-1 as float
    """
    coordinates = (
        tf.cast(coordinates, dtype=tf.dtypes.float64)
        / (w_factor, h_factor)
        / input_size
    )
    """
    _coordinates = (
        tf.cast(coordinates, dtype=tf.dtypes.float64)
        / (shape[1], shape[0])
    )
    if task_mode=="train":
        return (image, _coordinates, visibility)
    elif task_mode=="test":
        return (image,tf.cast(coordinates, dtype=tf.dtypes.float32),bboxes)

@tf.function
def tf_single_stage_heatmaps(
    stddev_tensor: tf.Tensor,
    shape_tensor: tf.Tensor,
    joints: tf.Tensor,
):
    precision = tf.dtypes.float32

    """
    limbs = tf.constant([[0,1], 
                         [1,2],
                         [2,3],
                         [3,4],
                         [4,5],
                         [6,7],
                         [7,8],
                         [7,12],
                         [7,13],
                         [13,14],
                         [14,15],
                         [12,11],
                         [11,10],
                         [8,9]])
    """
    
    limbs = tf.constant([[0,1], 
                         [1,2],
                         [2,3],
                         [3,4],
                         [4,5],
                         [6,7],
                         [7,8],
                         [8,12],
                         [9,12],
                         [12,13],
                         [11,10],
                         [10,9]])
    
    segments = tf.map_fn(
        fn=(
            lambda limb: tf_generate_segment(
                limb[0],limb[1],joints
            )
        ),
        elems = limbs,
        dtype=precision,
        parallel_iterations=10
    )

    heatmaps = tf.map_fn(
        fn=(
            lambda joint: tf_bivariate_normal_pdf(
                joint[:2],joint[2], stddev_tensor, shape_tensor, precision=precision
            )
            #if joint[2] == 1.0
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=precision)
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=tf.dtypes.uint8)
        ),
        elems=joints,
        dtype=precision,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )
    heatmaps = tf.transpose(heatmaps, [1, 2, 0])
    
    #"""
    limb_hms = tf.map_fn(
        fn=(
            lambda segment: tf_bivariate_segment_normal_pdf(
                segment[0:2,:],segment[2,:],stddev_tensor,shape_tensor, precision=precision
            )
        ),
        elems=segments,
        dtype=precision,
        parallel_iterations=10,
    )
    
    limb_hms = tf.transpose(limb_hms,[1,2,0])

    hms = tf.concat([heatmaps,limb_hms],axis=-1)
    #"""

    return hms #heatmaps

@tf.function
def tf_train_map_heatmaps(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    output_size: int = 64,
    stddev: float = 10.0,
    stacks: int = 3,
    scale_factor: float = 2.0,
) -> tf.Tensor:
    """Fourth step tf.data.Dataset mapper to generate heatmaps.

    This mapper is used on Training phase only.
    It would suit not Preditction phase since you need to have prior
    knowledge of the person position

    Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        coordinates (tf.Tensor): 2D Coordinate tensor(tf.dtypes.int32)
        visibility (tf.Tensor): 1D Visibility tensor(tf.dtypes.int32)
        output_size (int, optional): Heatmap shape. Defaults to 64
        stddev (float, optional): Standard deviation for Bivariate normal PDF.
            Defaults to 10.

    Returns:
        tf.Tensor: _description_
    """
    stack_tensor = tf.cast(tf.constant(stacks-1)-tf.range(0,stacks),dtype=tf.float32)
    bases = tf.cast(tf.convert_to_tensor([scale_factor]*stacks),dtype=tf.float32)
    stddev_tensor = tf.cast(tf.constant(stddev),dtype=tf.float32)*tf.pow(bases,stack_tensor)

    precision = tf.dtypes.float32
    # We move from relative coordinates to absolute ones by
    # multiplying the current coordinates [0-1] by the output_size
    new_coordinates = tf.cast(coordinates * tf.cast(output_size, dtype=tf.float64),precision)
    new_coordinates = tf.cast(new_coordinates,dtype=precision)
    visibility = tf.cast(tf.reshape(visibility, (-1, 1)), dtype=precision)

    # First we concat joint coordinate and visibility
    # to have a [NUN_JOINTS, 3] tensor
    # 0: X coordinate
    # 1: Y coordinate
    # 2: Visibility boolean as numeric
    joints = tf.concat([new_coordinates, visibility], axis=1)
    # We compute intermediate tensors
    shape_tensor = tf.cast([output_size, output_size], dtype=precision)
    
    #stddev_tensor = tf.cast([stddev, stddev], dtype=precision)
    # We generate joint's heatmaps
    # tf_bivariate_normal_pdf
    # tf_hm_distance

    ms_heatmaps = tf.map_fn(
        fn=lambda _stddev: tf_single_stage_heatmaps(_stddev,shape_tensor,joints),
        elems=stddev_tensor,
        dtype=precision,
        parallel_iterations=10
    )

    """
    heatmaps0 = tf.map_fn(
        fn=(
            lambda joint: tf_bivariate_normal_pdf(
                joint[:2],joint[2], stddev_tensor*4, shape_tensor, precision=precision
            )
            #if joint[2] == 1.0
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=precision)
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=tf.dtypes.uint8)
        ),
        elems=joints,
        dtype=precision,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )
    heatmaps0 = tf.transpose(heatmaps0, [1, 2, 0])

    heatmaps1 = tf.map_fn(
        fn=(
            lambda joint: tf_bivariate_normal_pdf(
                joint[:2],joint[2], stddev_tensor*2, shape_tensor, precision=precision
            )
            #if joint[2] == 1.0
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=precision)
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=tf.dtypes.uint8)
        ),
        elems=joints,
        dtype=precision,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )
    heatmaps1 = tf.transpose(heatmaps1, [1, 2, 0])


    heatmaps2 = tf.map_fn(
        fn=(
            lambda joint: tf_bivariate_normal_pdf(
                joint[:2],joint[2], stddev_tensor, shape_tensor, precision=precision
            )
            #if joint[2] == 1.0
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=precision)
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=tf.dtypes.uint8)
        ),
        elems=joints,
        dtype=precision,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )
    heatmaps2 = tf.transpose(heatmaps2, [1, 2, 0])

    heatmaps = tf.stack([heatmaps0,heatmaps1,heatmaps2],axis=0)
    """
    #_sum = tf.reduce_sum(heatmaps,axis=[0,1,2])
    #if _sum > 10:
    #    print("[IMPORTANT] Greater than 10")
    #"""
    print("      HEAT  () MAP SIZE: ....... ",ms_heatmaps)
    # We Transpose Heatmaps dimensions to have [HEIGHT, WIDTH, CHANNELS] data format
    
    return (image, ms_heatmaps)

@tf.function
def tf_train_map_normalize(
    image: tf.Tensor, heatmaps: tf.Tensor, normalization: str = None
) -> tf.Tensor:
    """Fifth step tf.data.Dataset mapper to normalize data.

    This mapper is used on Training phase only.
    It would suit not Preditction phase since you need to have prior
    knowledge of the person position

    Notes:
        The normalization methods are the following:
        - `ByMax`: Will constraint the Value between 0-1 by dividing by the global maximum
        - `L2`: Will constraint the Value by dividing by the L2 Norm on each channel
        - `Normal`: Will apply (X - Mean) / StdDev**2 to follow normal distribution on each channel

        Additional methodology involve:
        - `FromZero`: Origin is set to 0 maximum is 1 on each channel
        - `AroundZero`: Values are constrained between -1 and 1

    Additional Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        heatmaps (tf.Tensor): 3D Heatmap tensor(tf.dtypes.int32)
        normalization (str, optional): Normalization method. Defaults to None

    Returns:
        tf.Tensor: _description_
    """
    precision = tf.dtypes.float32
    #precision = tf.dtypes.int16

    image = tf.cast(image, dtype=precision)
    heatmaps = tf.cast(heatmaps, dtype=tf.float32)

    if normalization is None:
        pass
    if "Idem" in normalization:
        image = tf.cast(image,dtype=tf.float32)
        image = tf.cast(tf.math.divide_no_nan(
            image,1),dtype=tf.float32)
        heatmaps = tf.cast(tf.math.divide_no_nan(
            heatmaps,1),dtype=tf.float32)
    if "Normal" in normalization:
        #image = tf.math.divide_no_nan(
        #    image - tf.reduce_mean(image, axis=[0, 1]),
        #    tf.math.reduce_variance(image, axis=[0, 1]),
        #)
        image = tf.math.divide_no_nan(
            image - tf.reshape(tf.reduce_mean(image, axis=[0, 1]),[1,1,-1]),
            tf.reshape(tf.math.reduce_variance(image, axis=[0, 1]),[1,1,-1])
        )
        #heatmaps = tf.math.divide_no_nan(
        #    heatmaps - tf.reduce_mean(heatmaps, axis=[0, 1]),
        #    tf.math.reduce_variance(heatmaps, axis=[0, 1]),
        #)
    if "ByMax" in normalization:
        image = tf.math.divide_no_nan(
            image,
            255.0,
        )
        heatmaps = 1.0*tf.cast(heatmaps,dtype=tf.float32)
        #heatmaps = tf.math.divide_no_nan(
        #    heatmaps,
        #    tf.reduce_max(heatmaps),
        #)
    if "L2" in normalization:
        image = tf.linalg.l2_normalize(image, axis=[0, 1])
        heatmaps = tf.linalg.l2_normalize(heatmaps, axis=[0, 1])
    if "FromZero" in normalization:
        image = tf.math.divide_no_nan(
            image - tf.reduce_min(image, axis=[0, 1]),
            tf.reduce_max(image, axis=[0, 1]),
        )
        heatmaps = tf.math.divide_no_nan(
            heatmaps - tf.reduce_min(heatmaps, axis=[0, 1]),
            tf.reduce_max(heatmaps, axis=[0, 1]),
        )
    if "AroundZero" in normalization:
        #div = tf.constant(1/255.0)
        image = 2*tf.math.scalar_mul(1/255.0,image)-1.0
        heatmaps = 1.0*tf.cast(heatmaps,dtype=tf.float32)
        #tf.cast(tf.math.divide_no_nan(
        #    heatmaps,1),dtype=tf.float32)
        #heatmaps = 2 * (
        #    tf.math.divide_no_nan(
        #        heatmaps - tf.reduce_min(heatmaps, axis=[0, 1]),
        #        tf.reduce_max(heatmaps, axis=[0, 1]),
        #    )
        #    - 0.5
        #)
    return (image, heatmaps)

@tf.function
def tf_train_map_stacks(image: tf.Tensor, heatmaps: tf.Tensor, stacks: int = 1):
    """Sixth step tf.data.Dataset mapper to generate stacked hourglass.

    This mapper is used on Training phase only.
    It would suit not Preditction phase since you need to have prior
    knowledge of the person position

    Additional Notes:
        This function is build in compliance with `HTFDatasetHandler`.
        On a custom DatasetHandler this function might not suit your needs.
        See Dataset Documentation for more details

    Args:
        image (tf.Tensor): 3D Image tensor(tf.dtypes.int32)
        heatmaps (tf.Tensor): 3D Heatmap tensor(tf.dtypes.int32)
        stacks (int, optional): Number of heatmap replication . Defaults to 1

    Returns:
        tf.Tensor: _description_
    """
    # We apply the stacking
    heatmaps = tf_stack(heatmaps, stacks)
    return (image, heatmaps)
