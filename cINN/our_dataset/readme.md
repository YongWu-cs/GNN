The code structure is not very good due to timing issues.Due to training time constraints, there is no good API for direct calls.

# process pipelin
1. Calculate the mean and standard deviation of the Lab channel of the data using cal_offset_scal.py
2. Use instance_config to set the parameters of the instance model and train the model
3. Use instance_eval.py to evaluate the model.
4. Use create_ab_partition_base_instance.py and create_ab_partition to create instance-shaded global images and preserve instance-shaded global images.
5. Modify config.py to set the global model
6. Use train.py and eval.py to train and evaluate models
7. Use style_generation.py to generate colorization based on reference image.
