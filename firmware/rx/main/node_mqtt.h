#pragma once

/* Connects to Wi-Fi, syncs the clock and publishes detector events/status/metrics over MQTT with TLS (step 4.6). */
void node_mqtt_start(void);
