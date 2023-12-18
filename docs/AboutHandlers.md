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
    - generate_datasets(): Create `tf.data.Dataset` instances for the train, test, validation sets. Each `tf.data.Dataset` contains the cropped images (according the bounding boxes, _This has to do with the MPII Dataset_), heatmaps for each joint/landmark.



