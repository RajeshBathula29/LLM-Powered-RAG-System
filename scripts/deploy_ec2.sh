#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_ec2.sh  —  Bootstrap an Amazon Linux 2023 EC2 instance
#
# Run once on a fresh t3.medium (or larger) EC2 instance:
#   chmod +x deploy_ec2.sh && ./deploy_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  RAG System — EC2 Bootstrap"
echo "═══════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages …"
sudo dnf update -y
sudo dnf install -y git docker python3-pip curl

# ── 2. Docker + Compose ───────────────────────────────────────────────────────
echo "[2/6] Starting Docker …"
sudo systemctl enable docker --now
sudo usermod -aG docker ec2-user

# Docker Compose v2 (plugin)
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "[3/6] Cloning repository …"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/rag-system.git}"
git clone "$REPO_URL" /home/ec2-user/rag-system
cd /home/ec2-user/rag-system

# ── 4. Environment file ───────────────────────────────────────────────────────
echo "[4/6] Writing .env …"
cat > .env <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}
FLASK_ENV=production
EOF

# ── 5. Build + launch ─────────────────────────────────────────────────────────
echo "[5/6] Building and starting containers …"
# newgrp runs the rest of the script in the docker group without re-login
newgrp docker <<'DOCKERCMD'
cd /home/ec2-user/rag-system
docker compose build --no-cache
docker compose up -d
DOCKERCMD

# ── 6. Health check ───────────────────────────────────────────────────────────
echo "[6/6] Waiting for API to be healthy …"
for i in $(seq 1 12); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health || true)
  if [ "$STATUS" = "200" ]; then
    echo "✅  API is up!  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
    break
  fi
  echo "   Attempt $i/12 — waiting 5 s …"
  sleep 5
done

echo ""
echo "Grafana dashboard → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
echo "  username: admin  |  password: admin  (change immediately!)"
echo ""
echo "Done. EC2 instance is live."
