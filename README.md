# Jetson Nano ALPR Production Service

GPU-accelerated Automatic License Plate Recognition (ALPR) service for NVIDIA Jetson Nano. Designed for integration with IP cameras (Hikvision, etc.) via FTP triggers.

## Features

- **GPU Acceleration**: ~100ms inference using TensorRT (10 FPS)
- **Production Ready**: Docker-based deployment with auto-restart
- **SD Card Protection**: Uses `/dev/shm` (RAM) for temporary files
- **FTP Integration**: Receives images from cameras, uploads annotated results
- **Automatic Annotation**: Draws bounding boxes and plate text on images

## Performance

| Metric | Value |
|--------|-------|
| Inference Time | ~100ms |
| FPS | ~10 |
| Detection Accuracy | 84%+ |
| OCR Accuracy | 97%+ |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ionut13gmail/jetson-alpr.git
cd jetson-alpr

# Run setup wizard (creates .env, optionally installs vsftpd)
./run_alpr.sh setup
```

### 2. Configure

```bash
# Edit configuration
nano .env
```

### 3. Start Service

```bash
./run_alpr.sh start
```

## Management Script

The `run_alpr.sh` script provides easy management:

| Command | Description |
|---------|-------------|
| `./run_alpr.sh setup` | First-time setup wizard |
| `./run_alpr.sh start` | Start the service |
| `./run_alpr.sh stop` | Stop the service |
| `./run_alpr.sh restart` | Restart the service |
| `./run_alpr.sh update` | Pull from GitHub & rebuild |
| `./run_alpr.sh logs` | View service logs |
| `./run_alpr.sh status` | Show service status |
| `./run_alpr.sh test` | Test with sample image |

## Manual Build and Run

```bash
# Build the Docker image
sudo docker build -f Dockerfile.production -t alpr-service:latest .

# Create watch directories in RAM
sudo mkdir -p /dev/shm/alpr_inbox /dev/shm/alpr_outbox

# Run the service
sudo docker run -d \
  --name alpr-service \
  --runtime nvidia \
  --restart unless-stopped \
  -v /dev/shm/alpr_inbox:/watch/inbox \
  -v /dev/shm/alpr_outbox:/watch/outbox \
  -v $(pwd)/models:/app/models:ro \
  -e ALPR_FTP_HOST=192.168.100.238 \
  -e ALPR_FTP_USER=photo \
  -e ALPR_FTP_PASS=photo \
  alpr-service:latest
```

### 3. Configure Hikvision Camera FTP

Configure your camera to upload images on motion detection:

1. Go to **Configuration → Network → Advanced Settings → FTP**
2. Set:
   - Server Address: `<jetson-ip>` (e.g., `10.8.1.2`)
   - Port: `21`
   - User Name: `ftpcamera`
   - Password: `camera123`
   - Directory Structure: `Root directory`
3. Go to **Configuration → Event → Basic Event → Motion Detection**
4. Enable and link to FTP upload

## Deployment Options

### Option A: Standalone with Host vsftpd (Recommended)

Best performance with host-based FTP server:

```bash
# 1. Install vsftpd on Jetson host
sudo apt-get update
sudo apt-get install -y vsftpd

# 2. Create FTP user that writes to RAM
sudo useradd -m -d /dev/shm/alpr_inbox -s /usr/sbin/nologin ftpcamera
echo "ftpcamera:camera123" | sudo chpasswd

# 3. Configure vsftpd
sudo tee /etc/vsftpd.conf << 'EOF'
listen=YES
listen_ipv6=NO
anonymous_enable=NO
local_enable=YES
write_enable=YES
local_umask=022
chroot_local_user=YES
allow_writeable_chroot=YES
pasv_enable=YES
pasv_min_port=21100
pasv_max_port=21110
pasv_address=10.8.1.2
userlist_enable=YES
userlist_file=/etc/vsftpd.userlist
userlist_deny=NO
EOF

# 4. Create user whitelist
echo "ftpcamera" | sudo tee /etc/vsftpd.userlist

# 5. Create inbox directory (survives reboot via systemd)
sudo mkdir -p /dev/shm/alpr_inbox
sudo chown ftpcamera:ftpcamera /dev/shm/alpr_inbox

# 6. Restart vsftpd
sudo systemctl restart vsftpd
sudo systemctl enable vsftpd

# 7. Create directory on boot (add to rc.local or use systemd)
sudo tee /etc/systemd/system/alpr-dirs.service << 'EOF'
[Unit]
Description=Create ALPR directories in RAM
Before=vsftpd.service docker.service

[Service]
Type=oneshot
ExecStart=/bin/mkdir -p /dev/shm/alpr_inbox /dev/shm/alpr_outbox
ExecStart=/bin/chown ftpcamera:ftpcamera /dev/shm/alpr_inbox
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable alpr-dirs.service

# 8. Run ALPR service
docker-compose -f docker-compose.standalone.yml up -d
```

### Option B: Full Stack with Docker Compose

Includes FTP server in Docker:

```bash
# Set your Jetson's IP for passive FTP
export JETSON_IP=10.8.1.2

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f alpr-service
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPR_WATCH_DIR` | `/watch/inbox` | Directory to watch for new images |
| `ALPR_OUTPUT_DIR` | `/watch/outbox` | Temporary output directory |
| `ALPR_FTP_HOST` | `192.168.100.238` | Destination FTP server |
| `ALPR_FTP_USER` | `photo` | FTP username |
| `ALPR_FTP_PASS` | `photo` | FTP password |
| `ALPR_FTP_DIR` | `/` | Remote FTP directory |
| `ALPR_CONF_THRESH` | `0.4` | Detection confidence threshold |
| `ALPR_DELETE_PROCESSED` | `true` | Delete files after processing |
| `ALPR_LOG_LEVEL` | `INFO` | Logging level |

### Custom Configuration with .env file

```bash
# Create .env file
cat > .env << 'EOF'
ALPR_FTP_HOST=192.168.100.238
ALPR_FTP_USER=photo
ALPR_FTP_PASS=photo
ALPR_CONF_THRESH=0.5
ALPR_LOG_LEVEL=INFO
EOF

# Run with custom config
docker-compose -f docker-compose.standalone.yml up -d
```

## Alternative: Systemd Service (No Docker)

Run directly on host without Docker:

```bash
# Create systemd service
sudo tee /etc/systemd/system/alpr.service << 'EOF'
[Unit]
Description=ALPR License Plate Recognition Service
After=network.target vsftpd.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/john/fast-alpr
Environment=ALPR_WATCH_DIR=/dev/shm/alpr_inbox
Environment=ALPR_OUTPUT_DIR=/dev/shm/alpr_outbox
Environment=ALPR_FTP_HOST=192.168.100.238
Environment=ALPR_FTP_USER=photo
Environment=ALPR_FTP_PASS=photo
ExecStartPre=/bin/mkdir -p /dev/shm/alpr_inbox /dev/shm/alpr_outbox
ExecStart=/usr/bin/python3 /home/john/fast-alpr/alpr_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable alpr
sudo systemctl start alpr

# Check status
sudo systemctl status alpr
journalctl -u alpr -f
```

## Testing

### Manual Test

```bash
# Copy a test image to watch directory
cp assets/test_image.png /dev/shm/alpr_inbox/

# Check logs
docker logs -f alpr-service
```

### Expected Output

```
2024-01-15 10:30:45 [INFO] Processing: test_image.png
2024-01-15 10:30:45 [INFO]   Plate: DB20BDI (conf: 96.7%)
2024-01-15 10:30:45 [INFO] Uploaded to FTP: 20240115_103045_DB20BDI_test_image.jpg
```

## Output Format

Annotated images are named: `{timestamp}_{plate}_{original_name}.jpg`

Example: `20240115_103045_DB20BDI_motion_001.jpg`

The annotation includes:
- Green bounding box around detected plate
- Plate text and confidence percentage above the box

## Troubleshooting

### Service not starting

```bash
# Check Docker logs
docker logs alpr-service

# Verify GPU access
docker run --rm --runtime nvidia dustynv/l4t-ml:r32.7.1 nvidia-smi
```

### No plates detected

- Ensure image quality is sufficient
- Try lowering `ALPR_CONF_THRESH` (default 0.4)
- Check image format (JPG, PNG supported)

### FTP upload fails

```bash
# Test FTP connection
curl -v ftp://photo:photo@192.168.100.238/

# Check network
ping 192.168.100.238
```

## Directory Structure

```
jetson-alpr/
├── JETSON_DEPLOYMENT.md        # This deployment guide
├── README.md                   # Original fast-alpr README
├── jetson_alpr.py              # Core ALPR engine (Python 3.6 compatible)
├── alpr_service.py             # Production service with FTP
├── Dockerfile.production       # Production Docker image
├── Dockerfile.jetson-gpu       # Development/testing image
├── docker-compose.yml          # Full stack with FTP server
├── docker-compose.standalone.yml # ALPR service only
├── models/
│   ├── detector_opset15.onnx   # YOLO plate detector (opset 15)
│   └── ocr_opset15.onnx        # OCR model (opset 15)
└── assets/
    └── test_image.png          # Test image
```

## Hardware Requirements

- NVIDIA Jetson Nano (2GB or 4GB)
- JetPack 4.6.x
- microSD card (32GB+ recommended)
- Ethernet/WiFi connection

## Why /dev/shm?

Using `/dev/shm` (tmpfs/RAM filesystem) provides:
- **No SD card wear**: All temporary files in RAM
- **Fast I/O**: RAM is much faster than SD cards
- **Auto cleanup**: Files cleared on reboot
- **Size limited**: Default ~500MB on Jetson Nano (half of RAM)

The service is designed to process and delete files immediately, so storage usage stays minimal.
