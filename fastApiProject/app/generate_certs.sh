#!/bin/bash
# Генерация самоподписанного SSL-сертификата (два шага — надёжнее)
# Использование: bash generate_certs.sh
#
# Если openssl не найден / снова только key.pem — используйте:
#   python generate_certs.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="$SCRIPT_DIR/certs"
mkdir -p "$CERTS_DIR"

KEY="$CERTS_DIR/key.pem"
CERT="$CERTS_DIR/cert.pem"

echo "📁 Папка: $CERTS_DIR"

if ! command -v openssl >/dev/null 2>&1; then
    echo "❌ openssl не найден в PATH."
    echo "   Запустите вместо этого:  python generate_certs.py"
    exit 1
fi

echo "🔧 OpenSSL: $(openssl version)"

# Удаляем старые файлы, чтобы не путаться
rm -f "$KEY" "$CERT"

# Шаг 1: только приватный ключ
echo "→ Генерация ключа..."
openssl genrsa -out "$KEY" 4096 2>/dev/null

if [[ ! -f "$KEY" ]]; then
    echo "❌ Не удалось создать key.pem"
    exit 1
fi
echo "  OK: key.pem"

# Шаг 2: сертификат из уже созданного ключа
echo "→ Генерация сертификата..."
openssl req -new -x509 \
  -key "$KEY" \
  -out "$CERT" \
  -days 365 \
  -subj "//C=RU/ST=SPb/L=Saint-Petersburg/O=GUAP/OU=NeuroOnco/CN=localhost" \
  2>/dev/null

if [[ ! -f "$CERT" ]]; then
    echo "❌ Не удалось создать cert.pem"
    echo "   Попробуйте:  python generate_certs.py"
    exit 1
fi
echo "  OK: cert.pem"

# Проверка содержимого
echo ""
echo "✅ Готово:"
echo "   key.pem  — $(wc -c < "$KEY") байт"
echo "   cert.pem — $(wc -c < "$CERT") байт"
echo ""
echo "Запуск сервера:  python main.py"
echo "Браузер:         https://localhost:8000"
python main.py