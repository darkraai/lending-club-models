import os
import pickle

import numpy as np

class MyPredictor(object):
  def __init__(self, model, preprocessor):
    self._model = model
    self._preprocessor = preprocessor

  def predict(self, instances, **kwargs):
    inputs = np.asarray(instances)
    preprocessed_inputs = self._preprocessor.preprocess(inputs)
    if kwargs.get('probabilities'):
      probabilities = self._model.predict_proba(preprocessed_inputs)[:,1]
      return probabilities.tolist()
    else:
      outputs = self._model.predict(preprocessed_inputs)
      return outputs.tolist()

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    preprocessor = pickle.load(open(preprocessor_path, 'rb'))

    return cls(model, preprocessor)
