set(srcs "rx_main.c" "detector_task.c" "node_mqtt.c")
set(embed "")
# The TLS material is only needed (and only embedded) when MQTT is enabled: see tools/provision_device.sh.
if(CONFIG_CSI_RX_MQTT)
    set(embed "${CMAKE_CURRENT_LIST_DIR}/../certs/ca.pem"
              "${CMAKE_CURRENT_LIST_DIR}/../certs/client.crt"
              "${CMAKE_CURRENT_LIST_DIR}/../certs/client.key")
endif()

idf_component_register(SRCS ${srcs}
                       REQUIRES common csi_model esp_wifi esp_netif esp_event nvs_flash driver esp_timer esp_hw_support
                       PRIV_REQUIRES mqtt esp_system
                       EMBED_TXTFILES ${embed})
