# Backend (step 4.7)

Mosquitto (mutual TLS, per-device ACL) -> Telegraf -> InfluxDB 2 -> Grafana, in `backend/docker-compose.yml`.
Checked end to end with the real containers and `mosquitto_pub` (not with a real device): own-topic publish works,
a foreign topic is dropped, a revoked certificate and a missing client certificate are refused, events/metrics/status
reach InfluxDB, the Grafana dashboard is provisioned and its queries run.

## Start

```
tools/provision_pki.sh ca
tools/provision_pki.sh server <broker hostname> [more names or IPs]   # names the devices connect to; add "mosquitto"
tools/provision_pki.sh device backend                                 # used by Telegraf
tools/provision_pki.sh device rx-01                                   # copies the files to firmware/rx/certs/
cp backend/.env.example backend/.env                                  # then change all passwords and the token
cd backend && docker compose up -d
```

Grafana: http://localhost:3000 (bound to localhost; put a reverse proxy with TLS in front for anything else),
dashboard "CSI sensors". InfluxDB keeps 90 days (`INFLUX_RETENTION`).

`tools/provision_pki.sh revoke <id>` updates `backend/certs/crl.pem`; restart Mosquitto to load it.
`pki/` holds the CA key and all device keys: it is not in git, keep an offline backup. Device keys are generated on
the computer here; for production generate them on the device (Stage 6.3).

## Access rules (`mosquitto/acl`)

- A device (certificate CN = id) writes only `sensors/<id>/{event,status,metrics}` and reads only `sensors/<id>/cmd`.
- `backend` reads `sensors/#` and writes `sensors/+/cmd`. Send a calibration:
  `mosquitto_pub ... -t sensors/rx-01/cmd -m calibrate` (with the `backend` certificate).
- No plain listener, no anonymous clients, TLS 1.2+.

## Dashboard

Sensor table (last status, a silent device drops out after 15 min), alarm count, link-lost count, alarm timeline,
event history, P(person), window energy (drift: should stay flat in an empty room), inference time, heap, stream
resets and dropped frames. Alerts (drift above a threshold, silent device) are Stage 5.4/6.5.

Telegraf stores arrival time (the device clock may be unsynchronized) and subscribes to events with a persistent
session (QoS 1), so a short Telegraf restart does not lose events published meanwhile.
