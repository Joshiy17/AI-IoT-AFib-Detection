import numpy as np
from tensorflow.keras.models import load_model

model = load_model('results/afib_cnn_model.h5')
test_seq = np.array([[0.13, 0.15, 0.18, 0.19, 0.16]])
test_seq = test_seq.reshape((1, 5, 1))
prob = model.predict(test_seq)[0][0]
print("AFib probability:", prob)
if prob > 0.5:
    print("Possible AFib detected.")
else:
    print("Normal rhythm.")
