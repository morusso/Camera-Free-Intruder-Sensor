#!/usr/bin/env bash
# Own CA, broker certificate and per-device certificates for mTLS (steps 4.6 / 6.3).
#
#   tools/provision_pki.sh ca                       create the CA (once)
#   tools/provision_pki.sh server <name> [name|ip...]  broker certificate valid for all given names (-> backend/certs/);
#                                                   clients check the name they connect to, e.g. "mosquitto" inside docker
#   tools/provision_pki.sh device <id>              device certificate (-> firmware/rx/certs/ for rx-*, pki/devices/<id>/)
#   tools/provision_pki.sh revoke <id>              revoke a device and regenerate the CRL (-> backend/certs/crl.pem)
#
# The CA private key stays in pki/ca (outside git, back it up offline). The device key is generated on the
# computer that runs this script and copied to the device build; for production, generate it on the device instead.
set -euo pipefail
export SAN="${SAN:-DNS:unused}"  # referenced by the config even when no server certificate is issued

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKI="$ROOT/pki"
CA="$PKI/ca"
BACKEND="$ROOT/backend/certs"
FW="$ROOT/firmware/rx/certs"
DAYS_CA=3650
DAYS_CERT=825

cfg() {
cat <<CFG
[ca]
default_ca = CA_default
[CA_default]
dir = $CA
database = $CA/index.txt
serial = $CA/serial
crlnumber = $CA/crlnumber
new_certs_dir = $CA/issued
certificate = $CA/ca.pem
private_key = $CA/ca.key
default_md = sha256
default_days = $DAYS_CERT
default_crl_days = 30
policy = policy_any
unique_subject = no
copy_extensions = none
[policy_any]
commonName = supplied
[req]
distinguished_name = dn
prompt = no
[dn]
CN = x
[v3_server]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = $SAN
[v3_client]
basicConstraints = CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
CFG
}

need_ca() { [ -f "$CA/ca.key" ] || { echo "no CA: run '$0 ca' first" >&2; exit 1; }; }

sign() { # sign <csr> <out.crt> <ext-section>
    cfg > "$CA/openssl.cnf"
    openssl ca -batch -config "$CA/openssl.cnf" -extensions "$3" -in "$1" -out "$2" -notext 2>/dev/null
}

gen_crl() {
    mkdir -p "$BACKEND"
    cfg > "$CA/openssl.cnf"
    openssl ca -config "$CA/openssl.cnf" -gencrl -out "$BACKEND/crl.pem" 2>/dev/null
}

case "${1:-}" in
ca)
    [ ! -f "$CA/ca.key" ] || { echo "CA already exists in $CA" >&2; exit 1; }
    mkdir -p "$CA/issued" "$BACKEND"
    : > "$CA/index.txt"; echo 1000 > "$CA/serial"; echo 1000 > "$CA/crlnumber"
    umask 077
    openssl ecparam -name prime256v1 -genkey -noout -out "$CA/ca.key"
    openssl req -new -x509 -key "$CA/ca.key" -days $DAYS_CA -subj "/CN=CSI sensor CA" -out "$CA/ca.pem" \
        -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign"
    cp "$CA/ca.pem" "$BACKEND/ca.pem"
    gen_crl
    echo "CA created: $CA/ca.pem"
    ;;
server)
    need_ca; host="${2:?hostname}"; shift 2
    mkdir -p "$BACKEND"; umask 077
    openssl ecparam -name prime256v1 -genkey -noout -out "$BACKEND/server.key"
    openssl req -new -key "$BACKEND/server.key" -subj "/CN=$host" -out "$CA/server.csr"
    san="DNS:$host"
    for n in "$@"; do
        if [[ "$n" =~ ^[0-9.]+$ || "$n" == *:* ]]; then san="$san,IP:$n"; else san="$san,DNS:$n"; fi
    done
    SAN="$san" sign "$CA/server.csr" "$BACKEND/server.crt" v3_server
    echo "broker certificate for $san -> $BACKEND/"
    ;;
device)
    need_ca; id="${2:?device id}"
    [[ "$id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "device id: letters, digits, - and _ only" >&2; exit 1; }
    out="$PKI/devices/$id"; mkdir -p "$out"; umask 077
    openssl ecparam -name prime256v1 -genkey -noout -out "$out/client.key"
    openssl req -new -key "$out/client.key" -subj "/CN=$id" -out "$out/client.csr"
    sign "$out/client.csr" "$out/client.crt" v3_client
    cp "$CA/ca.pem" "$out/ca.pem"
    if [[ "$id" == rx-* ]]; then
        mkdir -p "$FW"; cp "$out"/{ca.pem,client.crt,client.key} "$FW/"
        echo "device $id -> $out/ and copied to $FW/ (the firmware embeds these at build time)"
    else
        echo "device $id -> $out/"
    fi
    ;;
revoke)
    need_ca; id="${2:?device id}"
    cfg > "$CA/openssl.cnf"
    openssl ca -config "$CA/openssl.cnf" -revoke "$PKI/devices/$id/client.crt" 2>/dev/null
    gen_crl
    echo "$id revoked; restart the broker (or send SIGHUP) to load $BACKEND/crl.pem"
    ;;
*)
    sed -n '2,12p' "$0"; exit 2
    ;;
esac
