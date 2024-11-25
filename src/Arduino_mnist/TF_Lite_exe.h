/*
 * Header-Datei für die Ausführung des TensorFlow Lite Modells
 * Diese Datei deklariert die Funktionen zum Setup und zur Ausführung des Modells.
 */

#include "mnist_model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Funktion zur Initialisierung des Modells
void setup_model();

// Funktion zur Ausführung des Modells mit Eingabedaten
float* model_execute(float *input_data);
