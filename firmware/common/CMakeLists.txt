# Shared by the TX and RX firmware: csi_link.h (header only) and the streaming CSI preprocessing and alarm logic (plain C, no IDF deps).
idf_component_register(SRCS "csi_preprocess.c" "csi_alarm.c" "csi_detector.c"
                       INCLUDE_DIRS "include")
