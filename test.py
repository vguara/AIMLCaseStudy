

ds = Dataset ("creditcard.csv")

ds.create_training (ratio = 0.8, label="Class", seed=None)

FR_list = [0.02, 0.05, 0.1, 0.15, 0.5]
ds.create_training_fraud(FR_list, seed=None)

