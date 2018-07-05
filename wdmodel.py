from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers


_CSV_COLUMNS  = [ 'EXT_SOURCE_2',
 'EXT_SOURCE_1',
 'EXT_SOURCE_3',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'OCCUPATION_TYPE',
 'TARGET']

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [''], [''],
                        [''],[''],[''],[0.0]]

_NUM_EXAMPLES = {
    'train': 39720,
    'validation': 9930,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def define_wide_deep_flags():
  """Add supervised learning flags, as well as wide-deep model type."""
  flags_core.define_base()
  flags_core.define_benchmark()

  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name="model_type", short_name="mt", default="wide_deep",
      enum_values=['wide', 'deep', 'wide_deep'],
      help="Select model topology.")

  flags_core.set_defaults(data_dir='/tmp/census_data',
                          model_dir='/tmp/census_model',
                          train_epochs=100,
                          epochs_between_evals=2,
                          batch_size=128)


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  EXT_SOURCE_2 = tf.feature_column.numeric_column('EXT_SOURCE_2')
  EXT_SOURCE_1 = tf.feature_column.numeric_column('EXT_SOURCE_1')
  EXT_SOURCE_3 = tf.feature_column.numeric_column('EXT_SOURCE_3')
  AMT_CREDIT = tf.feature_column.numeric_column('AMT_CREDIT')
  AMT_ANNUITY = tf.feature_column.numeric_column('AMT_ANNUITY')

  NAME_CONTRACT_TYPE = tf.feature_column.categorical_column_with_vocabulary_list(
      'NAME_CONTRACT_TYPE', [
          'Cash-loans', 'Revolving-loans'])

  OCCUPATION_TYPE= tf.feature_column.categorical_column_with_vocabulary_list(
      'OCCUPATION_TYPE', [
          'Laborers', 'Core-staff', 'Accountants', 'Managers', 'Drivers',
       'Sales-staff', 'Cleaning-staff', 'Cooking-staff',
       'Private-service-staff', 'Medicine-staff', 'Security-staff',
       'High-skill-tech-staff', 'Waiters-barmen-staff',
       'Low-skill-Laborers', 'Realty-agents', 'Secretaries', 'IT-staff',
       'HR-staff'])

  NAME_INCOME_TYPE= tf.feature_column.categorical_column_with_vocabulary_list(
      'NAME_INCOME_TYPE', [
         'Working', 'State-servant', 'Commercial-associate', 'Pensioner',
       'Unemployed', 'Student', 'Businessman', 'Maternity-leave'])

  NAME_FAMILY_STATUS = tf.feature_column.categorical_column_with_vocabulary_list(
      'NAME_FAMILY_STATUS', [
         'Single-not-married', 'Married', 'Civil-marriage', 'Widow',
       'Separated', 'Unknown'])

  # To show an example of hashing:
  NAME_HOUSING_TYPE = tf.feature_column.categorical_column_with_hash_bucket(
      'NAME_HOUSING_TYPE', hash_bucket_size=1000)

  # Transformations.
  EXT_SOURCE_3_buckets = tf.feature_column.bucketized_column(
      EXT_SOURCE_3, boundaries=[0,0.2,0.4,0.6,0.8,1.0])

  # Wide columns and deep columns.
  base_columns = [
      NAME_CONTRACT_TYPE , OCCUPATION_TYPE, NAME_INCOME_TYPE, NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,
      EXT_SOURCE_3_buckets ,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['NAME_CONTRACT_TYPE', 'NAME_HOUSING_TYPE'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [EXT_SOURCE_3_buckets, 'NAME_CONTRACT_TYPE', 'NAME_HOUSING_TYPE'], hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [EXT_SOURCE_2,EXT_SOURCE_1,EXT_SOURCE_3,AMT_CREDIT,AMT_ANNUITY,
      tf.feature_column.indicator_column(NAME_CONTRACT_TYPE),
      tf.feature_column.indicator_column(OCCUPATION_TYPE),
      tf.feature_column.indicator_column(NAME_INCOME_TYPE),
      tf.feature_column.indicator_column( NAME_FAMILY_STATUS),
      # To show an example of embedding
      tf.feature_column.embedding_column(NAME_HOUSING_TYPE, dimension=8),
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        linear_optimizer = tf.train.FtrlOptimizer(learning_rate=0.0001,l1_regularization_strength=0.005,l2_regularization_strength=0.001),
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(0.0001,initial_accumulator_value=0.1,l1_regularization_strength=0.005,l2_regularization_strength=0.001))

def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('TARGET')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def export_model(model, model_type, export_dir):
  wide_columns, deep_columns = build_model_columns()
  if model_type == 'wide':
    columns = wide_columns
  elif model_type == 'deep':
    columns = deep_columns
  else:
    columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_savedmodel(export_dir, example_input_fn)

def run_wide_deep(flags_obj):
  """Run Wide-Deep training and eval loop."""

  # Clean up the model directory if present
  shutil.rmtree(flags_obj.model_dir, ignore_errors=True)
  model = build_estimator(flags_obj.model_dir, flags_obj.model_type)

  train_file = os.path.join(flags_obj.data_dir, 'train_data_normalization')
  test_file = os.path.join(flags_obj.data_dir, 'validation_data_normalization')

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return input_fn(test_file,1, False, flags_obj.batch_size)

  run_params = {
      'batch_size': flags_obj.batch_size,
      'train_epochs': flags_obj.train_epochs,
      'model_type': flags_obj.model_type,
  }

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('wd_modeling', 'Census Income', run_params,
                                test_id=flags_obj.benchmark_test_id)

  loss_prefix = LOSS_PREFIX.get(flags_obj.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks, batch_size=flags_obj.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    tf.logging.info('Results at epoch %d / %d',
                    (n + 1) * flags_obj.epochs_between_evals,
                    flags_obj.train_epochs)
    tf.logging.info('-' * 60)

    for key in sorted(results):
      tf.logging.info('%s: %s' % (key, results[key]))

  #Export Trained Model for Serving
  wideColumns, DeepColumns = build_model_columns()
  feature_columns = DeepColumns
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
  export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  servable_model_dir = "/tmp/census_exported"
  servable_model_path = model.export_savedmodel(servable_model_dir, export_input_fn)
  print(" Done Exporting at Path - %s", servable_model_path)

def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_wide_deep(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_wide_deep_flags()
  absl_app.run(main)




