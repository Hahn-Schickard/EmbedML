/*
 * Arduino Demoprojekt für die Ausführung eines MNIST-Modells mit TensorFlow Lite.
 * Dieses Projekt zeigt, wie man ein TensorFlow Lite Modell auf einem Arduino-Board 
 * ausführt, um handgeschriebene Ziffern aus dem MNIST-Datensatz zu erkennen.
 */

// Einbindung der notwendigen Header-Dateien
#include <TensorFlowLite.h>
#include "input_data.h"
#include "Model_output.h"
#include "TF_Lite_exe.h"

// Setup-Funktion, die einmal beim Start ausgeführt wird
void setup() {
  // Initialisierung der seriellen Kommunikation zur Ausgabe von Ergebnissen
  Serial.begin(9600);
  // Setup des TensorFlow Lite Modells
  setup_model();
}

// Hauptschleife, die wiederholt ausgeführt wird
void loop() {
  // Schleife über die Eingabedaten (handgeschriebene Ziffern)
  for (int i = 0; i < 10; i++) {
    // Ausführen des Modells mit den Eingabedaten und Speichern der Vorhersagen
    float* output = model_execute(input_imgs[i]);
    
    // Finden des Indexes mit dem höchsten Wert (Vorhersage)
    int predicted_num = 0;
    float max_value = output[0];
    for (int j = 1; j < 10; j++) {
      if (output[j] > max_value) {
        max_value = output[j];
        predicted_num = j;
      }
    }
    
    // Ausgabe der Vorhersage über die serielle Schnittstelle
    Serial.print("Vorhergesagte Zahl: ");
    Serial.print(pred_labels[predicted_num]);
    Serial.print("\n");
  }

}
