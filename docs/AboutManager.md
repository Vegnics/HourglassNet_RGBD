# Manager
hourglass_tensorflow/handlers/__init\__.py

The `HTFManager` class is in charge data, dataset, and training configuration data validation. Besides, it invokes the different handlers in a specific order. If none of the handlers raise an error, the training process of the Stacked Hourglass Net is carried out.

## HTFManager(ObjectLogger):
 - Initialization: a `str` _filename_ is required (the filename must have any of these extensions: _.csv_, _.json_, _.toml_). The manager verifies the existence of the file _filename_, and parses the file to a `HTFConfig` object. It initializes a `HTFMetadata` instance, call it, and store it in the *_config* attribute.  

 - Properties: 
   - `HTFConfig` config: self._config
   - `HTFConfigMode` mode: self.config.mode
   - `Dict` VALIDATION_RULES
   - `HTFMetadata` metadata: self._metadata
 - Methods: 
   - **_import_object**(obj,config,metadata): It imports the object refered by the `HTFObjectReference` _obj_, and initializes it, by invoking the **init()** method with `HTFConfigField` _config_, `HTFMetadata` _metadata_. The initialized object is returned.
   - **__call\__**(): checks the VALIDATION_RULES, invokes the **train**() method.
   - **train()**: invoke the **_import_object**() method to create handlers for data, dataset, model, and train. The configuration info contained in the _\_config_ attribute is utilized to initialize the handlers. Finally, it calls the train handler to train the Hourglass network model.
