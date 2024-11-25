/*
 * Implementierung der Funktionen zum Setup und zur Ausführung des TensorFlow Lite Modells
 * Diese Datei enthält die Implementierungen der in der Header-Datei deklarierten Funktionen.
 */

#include "TF_Lite_exe.h"

namespace {
// Erstellen eines Speicherbereichs für Eingaben, Ausgaben und Zwischenarrays
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Globale Pointer für TensorFlow Lite Komponenten
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Array für die Vorhersagen
float* prediction = new float[10];
}

// Funktion zur Initialisierung des Modells
void setup_model() {
  // Initialisierung des Error Reporters
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Laden des TensorFlow Lite Modells
  model = tflite::GetModel(mnist_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Einbinden aller benötigten Operationen
  static tflite::AllOpsResolver resolver;

  // Erstellen eines Interpreters für das Modell
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Zuweisen von Speicher für die Tensoren des Modells
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Erhalten der Pointer auf die Eingabe- und Ausgabetensoren des Modells
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Funktion zur Ausführung des Modells mit Eingabedaten
float* model_execute(float *input_data) {
  // Kopieren der Eingabedaten in den Eingabetensor des Modells
  for (int i = 0; i < 784; ++i) {
    input->data.f[i] = *input_data;
    input_data++;
  }

  // Ausführen des Modells und Überprüfung auf Fehler
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Error by invoking interpreter\n");
    return 0;
  }

  // Lesen der Vorhersagen aus dem Ausgabetensor des Modells
  for (int i = 0; i < 10; i++) {
    prediction[i] = output->data.f[i];
  }

  return prediction;
}
