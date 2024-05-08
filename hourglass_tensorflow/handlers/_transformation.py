import tensorflow as tf
from typing import List
import cv2
import numpy as np

from hourglass_tensorflow.utils.tf import tf_stack
from hourglass_tensorflow.utils.tf import tf_load_image
from hourglass_tensorflow.utils.tf import tf_expand_bbox
from hourglass_tensorflow.utils.tf import tf_compute_bbox
from hourglass_tensorflow.utils.tf import tf_reshape_slice
from hourglass_tensorflow.utils.tf import tf_resize_tensor
from hourglass_tensorflow.utils.tf import tf_bivariate_normal_pdf
from hourglass_tensorflow.utils.tf import tf_generate_padding_tensor
from hourglass_tensorflow.utils.tf import tf_compute_padding_from_bbox
from hourglass_tensorflow.utils.tf import tf_rotate_tensor,tf_rotate_coords,tf_rotate_norm_coords, tf_hm_distance
from hourglass_tensorflow.utils.tf import tf_squarify_image_scale,tf_squarify_coordinates_scale,tf_expand_bbox_squared


@tf.function
def tf_train_map_build_slice(filename: tf.Tensor, coordinates: tf.Tensor) -> tf.Tensor:
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
    print(image)
    # Shape coordinates
    joints = tf_reshape_slice(coordinates, shape=3)
    # Extract coordinates and visibility from joints
    coordinates = joints[:, :2]
    visibility = joints[:, 2]
    return (image, coordinates, visibility,img_shape)

@tf.function
def tf_train_map_affine_augmentation(
    image: tf.Tensor,
    img_shape: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 64
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
    """
    affines = tf.constant([[-30,0.9],
                           [-30,1.0],
                           [-30,1.1],
                           [-15,0.9],
                           [-15,1.0],
                           [-15,1.1],
                           [0,0.9],
                           [0,1.0],
                           [0,1.1],
                           [15,0.9],
                           [15,1.0],
                           [15,1.1],
                           [25,0.9],
                           [25,1.0],
                           [25,1.1]],dtype=tf.float32)
    """
    affines = tf.constant([[-20,0.96],
                           [-18,0.96],
                           [-16,0.97],
                            [-12.0,0.97],
                            [-10.0,0.98],
                            [-8.0,1.0],
                           [0,1.0],
                           [0,1.0],
                           [0,1.0],
                           [12.0,0.97],
                            [10.0,0.98],
                            [8.0,1.0],
                           [20,0.96],
                           [18,0.96],
                           [16,0.97]],dtype=tf.float32)

    #angles = tf.constant([-25,-15,0,15,25])
    #scale = tf.constant([0.75,1.0,1.25],dtype=tf.float32)
    center = 0.5*(tf.cast(coordinates[2]+coordinates[3],dtype=tf.float32))
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
        dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )

    _coordinates = tf.map_fn(
        fn=(
            #lambda affine: tf_rotate_norm_coords(coordinates,
            lambda affine: tf_rotate_coords(coordinates,
                                            affine[0],
                                            affine[1],
            )
        ),
        elems=affines,
        dtype=tf.dtypes.float32,
        #dtype=tf.dtypes.float64,
        parallel_iterations=10,
    )
    _visibilities = tf.stack([visibility]*15, axis=0)

    _bboxf = tf.constant([1.095,1.08,1.08,1.07,1.05,1.09,1.03,1.08,1.13,1.095,1.08,1.08,1.07,1.05,1.09],
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
    print("ZIPPED",_zipped[0].shape,type(_zipped),type(_zipped[0]))
    _images = tf.reshape(tf.cast(_zipped[0],dtype=tf.uint8),[15,512,512,3])
    _coords = tf.reshape(_zipped[1],[15,16,2])
    _visibilities = tf.reshape(tf.cast(_zipped[2],dtype=tf.int32),[15,16])
    #return (image,coordinates,visibility)
    #return (_images,_coordinates,_visibilities)
    return (_images,_coords,_visibilities)

@tf.function
def tf_train_map_squarify_multiscale(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    bbox_enabled=False,
    bbox_factor=1.0,
) -> tf.Tensor:
    scales = tf.constant([1.0,1.25,1.5],dtype=tf.float64) 

    _images = tf.map_fn(
        fn = (lambda scale:tf_squarify_image_scale(
                                        image,
                                        coordinates,
                                        bbox_factor,
                                        scale,)
                                    ),
        elems=scales,
        dtype=tf.dtypes.uint8,
        parallel_iterations=10,)
    
    _coordinates = tf.map_fn(
        fn= (lambda scale: tf_squarify_coordinates_scale(
                                        coordinates,
                                        tf.shape(image),
                                        bbox_factor,
                                        scale,)
                                        ),
        elems=scales,
        dtype=tf.dtypes.int32,
        parallel_iterations=10)
    
    _visibilities = tf.stack([visibility]*3, axis=0)

    return (_images,_coordinates,_visibilities)

@tf.function
def tf_train_map_squarify_augmentation(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    bbox_enabled=False,
    bbox_factor=1.0,
) -> tf.Tensor:
    """
    Perform augmentation and squarify to a predefined shape (1024,1024)
    Rotate respect to 
    """
    affines = tf.constant([[-20,0.96],
                           [-18,0.96],
                           [-16,0.96],
                            [-12.0,0.96],
                            [-10.0,0.98],
                            [-8.0,1.0],
                           [0,0.95],
                           [0,1.0],
                           [0,1.1],
                           [12.0,0.96],
                            [10.0,0.98],
                            [8.0,1.0],
                           [20,0.96],
                           [18,0.96],
                           [16,0.96]],dtype=tf.float32)
    scales = tf.constant([1.0,1.25,1.5],dtype=tf.float64) 

    _images = tf.map_fn(
        fn = (lambda scale:tf_squarify_image_scale(
                                        image,
                                        coordinates,
                                        bbox_factor,
                                        scale,)
                                    ),
        elems=scales,
        dtype=tf.dtypes.uint8,
        parallel_iterations=10,)
    
    _coordinates = tf.map_fn(
        fn= (lambda scale: tf_squarify_coordinates_scale(
                                        coordinates,
                                        tf.shape(image),
                                        bbox_factor,
                                        scale,)
                                        ),
        elems=scales,
        dtype=tf.dtypes.int32,
        parallel_iterations=10)
    
    _visibilities = tf.stack([visibility]*3, axis=0)

    return (_images,_coordinates,_visibilities)
    

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
        bbox = tf_expand_bbox_squared(
            tf_compute_bbox(coordinates),
            tf.shape(image),
            bbox_factor=bbox_factor,
        )
    else:
        # Simulate a Bbox being the whole image
        shape = tf.shape(image)
        bbox = tf.cast([[0, 0], [shape[1] - 1, shape[0] - 1]])
    # Get Padding
    # Once the bbox is computed we compute
    # how much V/H padding should be applied
    # Padding is necessary to conserve proportions
    # when resizing
    padding = tf_compute_padding_from_bbox(bbox)
    # Generate Squared Image with Padding
    image = tf.pad(
        image[bbox[0, 1] : bbox[1, 1], bbox[0, 0] : bbox[1, 0], :],
        paddings=tf_generate_padding_tensor(padding),
    )
    nshape = tf.shape(image)
    scale = 512.0/tf.cast(nshape[0],dtype=tf.dtypes.float32)
    
    # Recompute coordinates
    # Given the padding and eventual bounding box
    # we need to recompute the coordinates from
    # a new origin
    
    #coordinates = coordinates - (bbox[0] - padding)
    
    coordinates = scale*(coordinates - tf.cast(bbox[0] - padding,dtype=tf.dtypes.float32))
    
    return (
        tf_resize_tensor(image,512),
        coordinates,
        visibility,
        visibility,
    )

@tf.function
def tf_train_map_resize_data(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    input_size: int = 256,
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
    image = tf.cast(tf_resize_tensor(image, size=input_size),dtype=tf.dtypes.uint8)
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
    coordinates = (
        tf.cast(coordinates, dtype=tf.dtypes.float64)
        / (shape[1], shape[0])
    )
    return (image, coordinates, visibility)

@tf.function
def tf_train_map_heatmaps(
    image: tf.Tensor,
    coordinates: tf.Tensor,
    visibility: tf.Tensor,
    output_size: int = 64,
    stddev: float = 10.0,
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
    precision = tf.dtypes.float32
    # We move from relative coordinates to absolute ones by
    # multiplying the current coordinates [0-1] by the output_size
    new_coordinates = coordinates * tf.cast(output_size, dtype=tf.float64)
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
    stddev_tensor = tf.cast([stddev, stddev], dtype=precision)
    # We generate joint's heatmaps
    # tf_bivariate_normal_pdf
    # tf_hm_distance
    heatmaps = tf.map_fn(
        fn=(
            lambda joint: tf_bivariate_normal_pdf(
                joint[:2], stddev_tensor, shape_tensor, precision=precision
            )
            if joint[2] == 1.0
            else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=precision)
            #else tf.zeros(tf.cast(shape_tensor, dtype=tf.dtypes.int32), dtype=tf.dtypes.uint8)
        ),
        elems=joints,
        dtype=precision,
        #dtype=tf.dtypes.uint8,
        parallel_iterations=10,
    )
    #_sum = tf.reduce_sum(heatmaps,axis=[0,1,2])
    #if _sum > 10:
    #    print("[IMPORTANT] Greater than 10")
    #"""
    print("      HEAT   MAP SIZE: ....... ",heatmaps)
    # We Transpose Heatmaps dimensions to have [HEIGHT, WIDTH, CHANNELS] data format
    heatmaps = tf.transpose(heatmaps, [1, 2, 0])

    return (image, heatmaps)

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
    #precision = tf.dtypes.float32
    precision = tf.dtypes.int16

    image = tf.cast(image, dtype=tf.uint8)
    heatmaps = tf.cast(heatmaps, dtype=tf.float32)

    if normalization is None:
        pass
    if "Idem" in normalization:
        image = tf.cast(tf.math.divide_no_nan(
            image,1),dtype=tf.uint8)
        heatmaps = tf.cast(tf.math.divide_no_nan(
            heatmaps,1),dtype=tf.float32)
    if "Normal" in normalization:
        image = tf.math.divide_no_nan(
            image - tf.reduce_mean(image, axis=[0, 1]),
            tf.math.reduce_variance(image, axis=[0, 1]),
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
        heatmaps = tf.math.divide_no_nan(
            heatmaps,
            tf.reduce_max(heatmaps),
        )
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
        image = 2 * (
            tf.math.divide_no_nan(
                image - tf.reduce_min(image, axis=[0, 1]),
                tf.reduce_max(image, axis=[0, 1]),
            )
            - 0.5
        )
        heatmaps = 2 * (
            tf.math.divide_no_nan(
                heatmaps - tf.reduce_min(heatmaps, axis=[0, 1]),
                tf.reduce_max(heatmaps, axis=[0, 1]),
            )
            - 0.5
        )
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
