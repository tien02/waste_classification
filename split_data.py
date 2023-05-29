import splitfolders
splitfolders.fixed("dataset", output="split_data",
    seed=40, fixed=(20, 20), oversample=False, group_prefix=None, move=False) # default values