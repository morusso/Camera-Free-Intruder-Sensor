idf_component_register(SRCS "csi_model.cpp" "model_data.c"
                       INCLUDE_DIRS "include"
                       REQUIRES esp_timer
                       PRIV_REQUIRES esp-tflite-micro)
