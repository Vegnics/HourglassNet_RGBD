# Handlers 

##  _HTFHandler(ABC,ObjectLogger)
*hourglass_tensorflow/handlers/meta.py*
- It is the fundamental handler class, all the handlers used in this project inherit its structure and methods. 
- **Initialization**:
  - **self._config**: Load a `HTFConfigField` object, containing the configuration for the handler.
  - **self._metadata**: Load a `HTFMetadata` object.
  - **self._executed** : Set a `boolean` attribute to `False`, meaning that the handler has not been called yet.
  - Invoke the **self.init_handler()** method with **args, **kargs. This method can be modified for any `Class` inherited from the fundamental handler according to its usage.
- **Base methods:**
  - **self.\__call__()**: Invoke the handler. This method is modified for any `Class` inherited from the fundamental handler. 
    1. It invokes the **self.run()** method.
    2. Set the **self.executed** property to `True`. 

## HTFDataHandler(_HTFDataHandler)

*hourglass_tensorflow/handlers/data.py*

 - `_HTFDataHandler()`:
    Prototype handler with the basic structure (properties,methods) for `HTFDataHandler`.
    - Properties:
      - config: self._config `HTFDataConfig` 
      - input_cfg: config.input `HTFDataInput`.
      - output_cfg: config.output `HTFDataOutput`. 
    - Methods:
        **self.run()** invokes the methods **self.prepare_input**, and **self.prepare_output**.
  - Methods:
    - **prepare_input()**: List the images in the image source folder.
    - **prepare_output()**: Validate the information related to the joints (indexing scheme, names, number), and read the joint location labels in the Output label source file. 
    - **get_data()**: Returns a `pd.Dataframe` containing the label headers of the label source file. 

## HTFDatasetHandler(_HTFDatasetHandler)

*hourglass_tensorflow/handlers/dataset.py*

- `_HTFDatasetHandler()`: Prototype handler for the `HTFDatasetHandler()` class.
  - Initialization:
    Set the following attributes.
    - self.data: Requires a `ndarray` or `pd.DataFrame`.
    - self.engine: select a `HTFEngine` according to the datatype of self.data.
  - Properties:
    - _engines: Returns a Dictionary of available engines of type `HTFEngine`.
    - config: self._config (a `HTFDatasetConfig` instance).
    - bbox: self.config.sets (a `HTFDatasetBBox` instance).
    - heatmap: self.config.sets (a `HTFDatasetHeatmap` instance).
  - Methods:
    - run(): run the _prepare_dataset()_, and _generate_datasets()_ methods.

- `HTFDatasetHandler()`: 
  - Properties: 
    - `bool` has_train: self.sets.train
    - `bool` has_test: self.sets.test
    - `bool` has_validation: self.sets.validation
    - `float` ratio_train: self.sets.ratio_train
    - `float` ratio_test: self.sets.ratio_test
    - `float` ratio_validation: self.sets.ratio_validation
  - Methods:
    - init_handler(): Initialize the test, train, validation datasets (`tf.data.Dataset`) as `None`.
    - prepare_dataset(): Split the main dataset (images + ground truth annotations) into train, test, validation sets (`ndarray` or `pd.DataFrame`).
    - generate_datasets(): Create `tf.data.Dataset` instances for the train, test, validation sets. Each `tf.data.Dataset` contains the cropped images (according the bounding boxes, _This has to do with the MPII Dataset_), heatmaps for each joint/landmark. The `tf.data.Dataset` objects are stored in the `HTFDatasetHandler()` object.

## HTFTrainHandler(_HTFTrainHandler)

*hourglass_tensorflow/handlers/train.py*

- `_HTFTrainHandler()`: Prototype class for the `HTFTrainHandler()`.
  - Initialization: 
    - Load the `HTFConfigField` config to _config. 
    - Set all the Class training parameters to `None` ( _epochs, _epoch_size, _batch_size, _learning_rate, _loss, _optimizer, _metrics, _callbacks).
    - call the **init_handler()** method.
  - Properties:
    - `HTFTrainConfig` config: self._config
  - Methods: 
    - **run()**: invoke the methods **compile()**, and **fit()**. 

- `HTFTrainHandler()`: 

  - Methods:
    - **init_handler**(): Set all the Class training parameters according to the `HTFTrainConfig` _config_ training configuration property.
    - **compile**(`keras.models.Model` model): compile a keras model according to the _config_ training configuration property ( _optimizer, _metrics, _loss).
    - **fit**(`keras.models.Model` model, `tf.data.Dataset` train_dataset, `tf.data.Dataset` test_dataset, `tf.data.Dataset` validation_dataset): Train a compiled keras model with the train, test, validation `Dataset`.
 > **Note**: The `HTFTrainHandler` contains only the configuration data related to the training parameters required to train a `keras.models.Model` with several `tf.data.Dataset`. It does not store any training data or model.

## Transformations (functions used by a `HTFDatasetHandler()`)
*hourglass_tensorflow/handlers/_transformation.py*

In total, a `HTFDatasetHandler()` instance performs 6 transformations to the raw training data, in order to generate the train, test, validation `tf.data.Dataset` instances.

- tf_train_map_build_slice(filename,coordinates): Receives an image filename and its coordinates data. Ouputs: image,coordinates,visibility.

- tf_train_map_squarify (image, coordinates,visibility): Crop the input image to have a square shape: Returns image (reshaped), coordinates (transformed according to the new image), visibility. 

- tf_train_map_resize_data (image, coordinates, visibility): Resizes the (previously cropped) image to a desired size, and modifies the coordinates to have a value between 0-1. 

- tf_train_map_heatmaps (image, coordinates, visibility, output_size, stddev): Generate the heatmaps from the coordinates. The size of the heatmaps is determined by the output_size. Whereas, the stddev determines the Bivariate normal PDF used to generate the heatmaps. The image is just passed. 

- tf_train_map_normalize (image, heatmaps, normalizations): The images and heatmaps are normalized according to a normalization method:
  - Normal: Values are centered at 0 with a std of 1. Negative and positive values.
  - ByMax: Values are divided by the maximum value. Taking values between 0-1.
  - L2: Values are divided by its L2 norm.
  - FromZero: The minimum value is set to zero. New values are all positives between 0-1.
  - AroundZero: The elements are linearly mapped to have values between [-1,1] 

- tf_train_map_stacks: Replicates the heatmaps for each Hourglass network. 
