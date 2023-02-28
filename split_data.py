import splitfolders
splitfolders.fixed("data/Garbage classification/Garbage classification", output="split_data",
    seed=42, fixed=(20, 20), oversample=False, group_prefix=None, move=False) # default values