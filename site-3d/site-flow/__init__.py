"""deepBlink for spot detection and localization.

Modules are arranged as follows:
- augment: Data augmentation to artificially increase dataset size.
- cli: Command line interface for inferencing.
- data: Data manipulation. Mainly to properly format for training.
- datasets: Unique data import functions.
- inference: Prediction related functions.
- io: File-manipulation-related functions.
- losses: Simple functions returning model losses.
- metrics: Quantitative output of training / model performance.
- models: Training loop containing classes for each type of model.
- networks: Architecture / building of model structure.
- optimizers: Simple functions returning model optimizers.
- training: Core training loop and callbacks.
- util: Basic utility functions not fitting into a category.
"""

# __version__ = "0.1.4"

# from . import config
# from . import dataset
# from . import log
# from . import loss
# from . import metirc
# from . import model
# from . import optimizer
# from . import utils