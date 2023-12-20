# Engines

An engine is in charge of receiving the training data (as a `ndarray`, `pd.DataFrame`, or `tf.Tensor`) and extract the selected `Set` of image filenames, filter parts of the training data, and obtain part of the data according to their columns --(NOT SURE YET WHAT A COLUMN MEANS UNDER THIS CONTEXT)--. The column names are specified in the `HTFMetadata` metadata.

## HTFEngine(ABC, ObjectLogger)
Base engine.
- Initialization:
  - Receives the `HTFMetadata` metadata.
- Properties:
  - FOR_TYPE : Specifies the datatype of the data. 
- Methods:
  - **get_images**(data,column): Returns the `Set` of images in the data. The column should refer to the column containing the image filename. 
  - **filter_data**(data,column,set_name): Return the data which specified column matches with the set_name.  
  - **select_subset_from_images**(data, image_set, column): Return the images whose columns have values that are included in the `Set[str]` image_set.
  - **get_columns**(data, columns): Obtain the data corresponding to the `List` columns. 
  - **to_list**(data): Convert the data to a `List`. Only for `ndarray` data. 

## HTFNumpyEngine(HTFEngine)

`HTFEngine` class tailored to deal with `ndarray` data.

## HTFPandasEngine(HTFEngine)

`HTFEngine` class tailored to deal with `pd.DataFrame` data.

## HTFTensorflowEngine(HTFEngine)

`HTFEngine` class tailored to deal with `tf.Tensor` data.