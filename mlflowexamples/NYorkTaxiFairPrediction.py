import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow.keras
import hashlib
import time
import random
import string
from tensorflow import keras
from sklearn import preprocessing


class NYorkTaxiFairPrediction:
    __slots__ = ['_params', '_data', '_learning_rate', '_steps', '_batch_size', '_dataset_size', '_model_dir',
                 '_activation_function', '_train_sample', '_validate_sample', '_test_sample', '_summary_steps', '_checkpoints_steps', '_throttle_secs', '_output_path', '_test_pd_df', '_train_labels', '_validation_labels', '_train_df_scaled', '_validation_df_scaled', '_test_scaled', '_estimator', '_matrics', '_model'
                 ]
    """
    General class for NYorkTaxiFairPrediction
    """
    train_pd_df = []
    validate_pd_df = []
    test_pd_df = []

    def __init__(self, *params, **data):
        """
        Constructor for the NYorkTaxiFairPrediction
        """
        self._params = params
        self._data = data
        self._test_pd_df = None
        self._train_labels = None
        self._validation_labels = None
        self._train_df_scaled = None
        self._validation_df_scaled = None
        self._test_scaled = None
        self._estimator = None
        self._matrics = None
        self._model = None

        for dictionary in params:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for dictionary in data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

    @classmethod
    def new_instance(cls, *params, **data):
        return cls(*params, **data)

    @property
    def params(self):
        """
        Getter for model parameters
        """
        return self._params

    @property
    def data(self):
        """
        Getter for data paths
        """
        return self._data

    @property
    def model(self):
        """
        Getter for data paths
        """
        return self._model

    @property
    def matrics(self):
        """
        Getter for data paths
        """
        return self._matrics

    # helper functions
    # Spark will write parquet files in to a folder with many file parts.
    # This helper function will help to combine all of these file parts to a Pandas dataframe
    @staticmethod
    def read_parquet_folder_as_pandas(path, verbosity=1):
        files = [f for f in os.listdir(path) if f.endswith("parquet")]

        if verbosity > 0:
            print("{} parquet files found. Beginning reading...".format(len(files)), end="")
            start = datetime.datetime.now()

        df_list = [pd.read_parquet(os.path.join(path, f)) for f in files]
        df = pd.concat(df_list, ignore_index=True)

        if verbosity > 0:
            end = datetime.datetime.now()
            print(" Finished. Took {}".format(end - start))
        return df

    # wrapper to call Tenserflow API with the list of feature columns and labels
    @staticmethod
    def input_function(features, labels=None, shuffle=False):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"raw_input": features},
            y=labels,
            shuffle=shuffle
        )
        return input_fn

    # helper function to save the predictions to the S3 bucket
    @staticmethod
    def save_output(df, prediction_df, prediction_column, file_path):
        df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
        df_with_key = df[['key', prediction_column]]
        df_with_key.name = 'New-York-taxi-fare-predictions.csv'
        df_with_key.to_csv(file_path, index=False)
        return file_path

    @staticmethod
    def get_hash():
        hash = hashlib.sha1()
        hash.update(str(time.time()).encode('utf-8'))
        return hash.hexdigest()[:15]

    @staticmethod
    def random_key(length):
        key = ''
        for i in range(length):
            key += random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        return key

    def load_scale_and_preprocess_data(self):
        # if the data is already in class variables take it from there
        # else read the data
        if not NYorkTaxiFairPrediction.train_pd_df:
            _train_pd_df = NYorkTaxiFairPrediction.read_parquet_folder_as_pandas(self._train_sample)
            NYorkTaxiFairPrediction.train_pd_df.append(_train_pd_df)
        else:
            _train_pd_df = NYorkTaxiFairPrediction.train_pd_df[0]

        if not NYorkTaxiFairPrediction.validate_pd_df:
            _validate_pd_df = NYorkTaxiFairPrediction.read_parquet_folder_as_pandas(self._validate_sample)
            NYorkTaxiFairPrediction.validate_pd_df.append(_validate_pd_df)
        else:
            _validate_pd_df = NYorkTaxiFairPrediction.validate_pd_df[0]

        if not NYorkTaxiFairPrediction.test_pd_df:
            _test_pd_df = NYorkTaxiFairPrediction.read_parquet_folder_as_pandas(self._test_sample)
            NYorkTaxiFairPrediction.test_pd_df.append(_test_pd_df)
        else:
            _test_pd_df = NYorkTaxiFairPrediction.test_pd_df[0]

        _train_labels = _train_pd_df['fare_amount'].values
        _validation_labels = _validate_pd_df['fare_amount'].values
        _test_labels = _test_pd_df['fare_amount'].values

        _train_features = _train_pd_df.drop(['fare_amount', 'key'], axis=1)
        _validation_features = _validate_pd_df.drop(['fare_amount', 'key'], axis=1)

        _test_dataset = _test_pd_df.drop(['passenger_count', 'key'], axis=1)

        scaler = preprocessing.MinMaxScaler()
        _train_df_scaled = scaler.fit_transform(_train_features).astype(np.float32)
        _validation_df_scaled = scaler.transform(_validation_features).astype(np.float32)
        _test_scaled = scaler.fit_transform(_test_dataset).astype(np.float32)

        self._test_pd_df = _test_pd_df
        self._train_labels = _train_labels
        self._validation_labels = _validation_labels
        self._train_df_scaled = _train_df_scaled
        self._validation_df_scaled = _validation_df_scaled
        self._test_scaled = _test_scaled

    def create_model(self, activation_func='relu', learning_rate=0.0001):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256, activation=activation_func, input_shape=(self._train_df_scaled.shape[1],), name='raw'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(128, activation=activation_func))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation=activation_func))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(32, activation=activation_func))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(16, activation=activation_func))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(1, name='predictions'))

        adam = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam, metrics=['mae'])
        self._model = model

    def train_and_evaluate(self):
        new_model_dir = self._model_dir + '/' + NYorkTaxiFairPrediction.get_hash()

        run_config = tf.estimator.RunConfig(model_dir=new_model_dir,
                                            save_summary_steps=5000,
                                            save_checkpoints_steps=self._checkpoints_steps)

        _estimator = keras.estimator.model_to_estimator(keras_model=self._model, config=run_config)

        train_spec = tf.estimator.TrainSpec(input_fn=
                                            NYorkTaxiFairPrediction.input_function(self._train_df_scaled,
                                                                                       self._train_labels, True),
                                            max_steps=self._steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=
                                          NYorkTaxiFairPrediction.input_function(self._validation_df_scaled,
                                                                                     self._validation_labels, True),
                                          steps=self._steps, throttle_secs=300)

        _matrics = tf.estimator.train_and_evaluate(_estimator, train_spec=train_spec, eval_spec=eval_spec)

        self._estimator = _estimator
        self._matrics = _matrics
        print("*" * 100)
        print(_matrics)

    def predict(self):
        prediction = self._estimator.predict(NYorkTaxiFairPrediction.input_function(self._test_scaled))
        prediction_df = pd.DataFrame(prediction)
        _predictions = prediction_df.to_numpy()
        NYorkTaxiFairPrediction.save_output(self._test_pd_df, prediction_df, 'fare_amount', self._output_path)

    def mlflow_run(self, r_name='test'):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run using the MLflow APIs
        :param r_name: Name of the run as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            self.load_scale_and_preprocess_data()
            self.create_model()
            self.train_and_evaluate()
            self.predict()

            _params = self.params[0]
            matrics = self._matrics[0]

            mae = matrics['mean_absolute_error']
            mse = matrics['loss']

            # compute  regression evaluation metrics
            rmse = np.sqrt(mse)
            # Log model and params using the MLflow APIs
            mlflow.log_params(_params)
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.keras.log_model(self.model, "Keras_model for NY-Taxi dataset")

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rmse)

            return (experimentID, runID)