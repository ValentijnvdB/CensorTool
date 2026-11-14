# HTTP server (`http`)

```
python main.py http [--host localhost] [--port 8443] [--ssl --ssl-cert certs/cert.pem --ssl-key certs/key.pem]
```

`https` works as an alias of `http`.

Starts a server with endpoints to censor images (`/censor_image`), videos (`/censor_video`, `/videos`) and a stream of
frames over a WebSocket (`/censor_frame`), and to only detect (`/detect_features`). The models are loaded once at
startup; a request can send its own censor config. This is what
[CensorExtension](https://github.com/ValentijnvdB/CensorExtension) uses.

The endpoints and their request and response formats are described in [http_api.md](../http_api.md). Server
settings such as file retention and the maximum video download size are in [config.md](../config.md#http-server).
Files uploaded to and produced by the server are kept in `data/server/`.

By default the server only accepts connections from this computer. Use `--host 0.0.0.0` to accept connections from
other devices.

## HTTPS

With `--ssl` the server uses HTTPS, with the certificate and key in `certs/cert.pem` and `certs/key.pem` (or the
files given with `--ssl-cert` and `--ssl-key`). For use on your own computer or network, a self-signed certificate
is enough:

```
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

Add the name or IP address you connect to (e.g. `IP:192.168.1.10`) to `subjectAltName` when the server is used from
other devices. Browsers don't trust a self-signed certificate: open `https://localhost:8443/health` once and accept
the warning. `python main.py validate --only http --ssl` checks that the certificate loads and warns when it has
expired; create a new one the same way.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `localhost` | Address to listen on. Use `0.0.0.0` to accept connections from other devices. |
| `--port` | `8443` | Port to listen on. |
| `--ssl` | off | Use HTTPS. |
| `--ssl-cert`, `--ssl-key` | `certs/cert.pem`, `certs/key.pem` | Certificate and key for HTTPS. |

The [general arguments](../../README.md#usage) (`--profile`, `--debug`, ...) work as well.
