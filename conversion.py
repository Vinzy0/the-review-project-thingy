import tensorflow as tf

h5_model_path = r"F:\Godot\Practice\the-review-project-thingy\cnn_model\hair_type_efficientnetb3_finetuned.h5" #Ignore the godot, I'm just using this as temporary path
model = tf.keras.models.load_model(h5_model_path, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
tflite_model_path = "hair_model.tflite"

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")
#I'll be back to fix this