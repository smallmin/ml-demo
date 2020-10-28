import tensorflow as tf
import tensorflow.keras as keras
from datasets import generate_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GRUCell, RNN, Dense, BatchNormalization
from plotting import plot_metric, plot_predictions
import tensorflow_addons as tfa
from tensorflow_addons.rnn import LayerNormLSTMCell

exercice_number = 2
print("exercice {}\n==================".format(exercice_number))
data_inputs, expected_outputs = generate_data(
    # See: https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/datasets.py
    exercice_number=exercice_number,
    n_samples=50000,
    window_size_past=None,
    window_size_future=None,
)

data_inputs.shape
expected_outputs.shape

sequence_length = data_inputs.shape[1]
input_dim = data_inputs.shape[2]
output_dim = expected_outputs.shape[2]


inputs = keras.Input(
    shape=(None, input_dim), dtype=tf.dtypes.float32, name="encoder_inputs"
)

encoder = RNN(
    cell=[LayerNormLSTMCell(128), LayerNormLSTMCell(128)],
    return_sequences=True,
    return_state=True,
)

encoder_output, *encoder_hidden = encoder(inputs)

context_vector, attention_weights = keras.layers.Attention()(
    [encoder_hidden, encoder_output]
)

x = tf.expand_dims(context_vector, axis=1)

decoder = RNN(
    cell=[LayerNormLSTMCell(128), LayerNormLSTMCell(128)],
    return_sequences=True,
    return_state=False,
)

replicated_last_encoder_output = tf.repeat(
    input=x, repeats=sequence_length, axis=1
)

decoder_outputs = decoder(
    replicated_last_encoder_output, initial_state=x
)

decoder_dense = Dense(output_dim)
outputs = decoder_dense(decoder_outputs)

model = keras.Model(inputs, outputs, name="seq2seq")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-2),
    loss=keras.losses.MeanSquaredError(),
    # loss=keras.losses.MeanAbsolutePercentageError(),
    metrics=["mae", "mape", "mse"],
)


history = model.fit(
    data_inputs, expected_outputs, batch_size=50, epochs=5, validation_split=0.1
)

best1 = model

plot_metric(history.history["mape"], history.history["val_mape"])
plot_metric(history.history["mse"], history.history["val_mse"])

test_inputs, test_expected_outputs = generate_data(
    # See: https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/datasets.py
    exercice_number=exercice_number,
    n_samples=1024,
    window_size_past=None,
    window_size_future=None,
)

result = model.evaluate(test_inputs, test_expected_outputs)

predicts = model.predict(test_inputs)

for i in range(5):
    plot_predictions(test_inputs[i], test_expected_outputs[i], predicts[i])
