import tensorflow as tf
import os
import numpy as np

exported_path = '/tmp/census_exported/1530173843'
predictionoutputfile = 'test_output.csv'
predictioninputfile = 'test_input'


def main():
    with tf.Session() as sess:
        # load the saved model
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)

        # get the predictor , refer tf.contrib.predictor
        predictor = tf.contrib.predictor.from_saved_model(exported_path)

        prediction_OutFile = open(predictionoutputfile, 'w')

        # Write Header for CSV file
        prediction_OutFile.write(
            "SK_ID_CURR,EXT_SOURCE_2, EXT_SOURCE_1,EXT_SOURCE_3,AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE,DAYS_BIRTH,DAYS_EMPLOYED,DAYS_REGISTRATION,DAYS_ID_PUBLISH,NAME_CONTRACT_TYPE,CODE_GENDER,NAME_TYPE_SUITE,NAME_INCOME_TYPE,NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,OCCUPATION_TYPE,TARGET")
        prediction_OutFile.write('\n')

        # Read file and create feature_dict for each record
        with open(predictioninputfile) as inf:
            # Skip header
            next(inf)
            for line in inf:
                # Read data, using python, into our features
                SK_ID_CURR, EXT_SOURCE_2, EXT_SOURCE_1, EXT_SOURCE_3, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, NAME_CONTRACT_TYPE, CODE_GENDER, NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE = line.strip().split(
                    ",")

                # Create a feature_dict for train.example - Get Feature Columns using
                feature_dict = {
                    'SK_ID_CURR': _float_feature(value=int(SK_ID_CURR)),
                    'EXT_SOURCE_2': _float_feature(value=float(EXT_SOURCE_2)),
                    'EXT_SOURCE_1': _float_feature(value=float(EXT_SOURCE_1)),
                    'EXT_SOURCE_3': _float_feature(value=float(EXT_SOURCE_3)),
                    'AMT_CREDIT': _float_feature(value=float(AMT_CREDIT)),
                    'AMT_ANNUITY': _float_feature(value=float(AMT_ANNUITY)),
                    'AMT_GOODS_PRICE': _float_feature(value=float(AMT_GOODS_PRICE)),
                    'DAYS_BIRTH': _float_feature(value=float(DAYS_BIRTH)),
                    'DAYS_EMPLOYED': _float_feature(value=float(DAYS_EMPLOYED)),
                    'DAYS_REGISTRATION': _float_feature(value=float(DAYS_REGISTRATION)),
                    'DAYS_ID_PUBLISH': _float_feature(value=float(DAYS_ID_PUBLISH)),
                    'NAME_CONTRACT_TYPE': _bytes_feature(value=NAME_CONTRACT_TYPE.encode()),
                    'CODE_GENDER': _bytes_feature(value=CODE_GENDER.encode()),
                    'NAME_TYPE_SUITE': _bytes_feature(value=NAME_TYPE_SUITE.encode()),
                    'NAME_INCOME_TYPE': _bytes_feature(value=NAME_INCOME_TYPE.encode()),
                    'NAME_EDUCATION_TYPE': _bytes_feature(value=NAME_EDUCATION_TYPE.encode()),
                    'NAME_FAMILY_STATUS': _bytes_feature(value=NAME_FAMILY_STATUS.encode()),
                    'NAME_HOUSING_TYPE': _bytes_feature(value=NAME_HOUSING_TYPE.encode()),
                    'OCCUPATION_TYPE': _bytes_feature(value=OCCUPATION_TYPE.encode()),

                }

                # Prepare model input

                model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))

                model_input = model_input.SerializeToString()
                output_dict = predictor({"inputs": [model_input]})

                print(" prediction Label is ", output_dict['classes'])
                print('Probability : ' + str(output_dict['scores']))

                # Positive label = 1
                prediction_OutFile.write(
                    str(SK_ID_CURR) + "," + str(EXT_SOURCE_2) + "," + str(EXT_SOURCE_1) + "," + str(
                        EXT_SOURCE_3) + "," + str(
                        AMT_CREDIT) + "," + str(AMT_ANNUITY) + "," + str(AMT_GOODS_PRICE) + "," + str(
                        DAYS_BIRTH) + "," + str(DAYS_EMPLOYED) + "," + str(DAYS_REGISTRATION) + "," + str(
                        DAYS_ID_PUBLISH) + "," + NAME_CONTRACT_TYPE + "," + CODE_GENDER + "," + NAME_TYPE_SUITE + "," + NAME_INCOME_TYPE + "," + NAME_EDUCATION_TYPE + "," + NAME_FAMILY_STATUS + "," + NAME_HOUSING_TYPE + "," + OCCUPATION_TYPE + ",")
                label_index = np.argmax(output_dict['scores'])
                prediction_OutFile.write(str(label_index))
                prediction_OutFile.write(',')
                prediction_OutFile.write(str(output_dict['scores'][0][label_index]))
                prediction_OutFile.write('\n')

    prediction_OutFile.close()


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    main()