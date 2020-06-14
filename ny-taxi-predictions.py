import sys
import mlflowexamples as me

if __name__ == '__main__':
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0001
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    input_data = {"_train_sample": '/dbfs/mnt/blogs_pl/taxi_fare_feature_eng_train_sample2',
                  "_validate_sample": '/dbfs/mnt/blogs_pl/taxi_fare_feature_eng_validate_sample2',
                  "_test_sample": '/dbfs/mnt/blogs_pl/taxi_fare_feature_eng_test_sample2'}

    input_params = {"_learning_rate": learning_rate , "_steps": 100000, "_batch_size": steps, "_dataset_size": 4000000,
                    "_model_dir": '/dbfs/tmp/models', "_activation_function": "relu",
                    "_checkpoints_steps": 5000, "_output_path": '/dbfs/mnt/blogs_pl/output1'}

    nyt = me.NYorkTaxiFairPrediction.new_instance(input_params, input_data)
    (experimentID, runID) = nyt.mlflow_run(me.NYorkTaxiFairPrediction.random_key(10))
    print(
        "MLflow Run for NYorkTaxiFairPrediction completed with run_id {} and experiment_id {}".format(runID, experimentID))
