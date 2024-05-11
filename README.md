# Shady NeRF

You can create the python environment used for running the code with the following command.

```sh
conda env create -f environment.yml
```

To run the training the following command can be used in the `lodnelf-model` directory.

```sh
python3 -m lodnelf.train.train_cmd --run_name "experiment_4" --config "SimpleRedCarModel" --model_save_dir "models/experiment_4" --data_dir "data"
```