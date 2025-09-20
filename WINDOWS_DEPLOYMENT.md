# 🚀 AI Conspects Server - Windows Deployment

## 📋 Краткое описание

Полнофункциональный сервер для iOS приложения AI Conspects с поддержкой:
- ✅ Аутентификация через device registration + JWT
- ✅ Загрузка и обработка аудиофайлов
- ✅ CRUD операции с заметками  
- ✅ AI чат с контекстом заметок
- ✅ SQLite база данных (легко мигрировать на PostgreSQL)
- ✅ Поддержка GPU (RTX 4090) для ускорения обработки

## 🏃‍♂️ Быстрый старт

### 1. Скачать и установить
```bash
git clone https://github.com/brainnotincluded/ai-conspects-server.git
cd ai-conspects-server
```

### 2. Создать виртуальное окружение
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установить зависимости
```bash
pip install fastapi uvicorn sqlalchemy python-jose[cryptography] passlib[bcrypt] python-multipart
```

### 4. Запустить сервер
```bash
# Простой запуск
start_server.bat

# Или вручную
python ios_compatible_server.py
```

### 5. Проверить работу
```bash
# Откройте браузер и перейдите на:
http://localhost:5000

# Или проверьте health endpoint:
curl http://localhost:5000/health
```

## 📱 iOS App Endpoints

Сервер предоставляет все необходимые endpoints для iOS приложения:

### 🔑 Аутентификация
- `GET /auth/generate-device-id` - Генерация device ID
- `POST /auth/register` - Регистрация устройства
- `GET /auth/me` - Информация о пользователе

### 🎙️ Обработка аудио  
- `POST /batches/upload-files` - Загрузка аудиофайлов
- `GET /batches/{batch_id}` - Статус обработки
- `GET /batches/{batch_id}/results` - Результаты обработки

### 📝 Заметки
- `GET /notes/` - Список заметок
- `GET /notes/{note_id}` - Конкретная заметка
- `POST /notes/{note_id}/toggle-favorite` - Добавить в избранное
- `GET /notes/popular-tags` - Популярные теги

### 🤖 AI Чат
- `POST /chat/query` - Отправка сообщения AI
- `POST /chat/study-plan` - Создание плана обучения

### ❤️ Системные
- `GET /health` - Проверка состояния сервера
- `GET /` - Информация о сервере

## 🔧 Конфигурация

### Базовые настройки
Все настройки автоматически задаются в `ios_compatible_server.py`:
- **Порт**: 5000 
- **База данных**: SQLite (`ai_conspects.db`)
- **JWT**: 24-часовые токены
- **CORS**: Разрешены все домены (для разработки)

### GPU поддержка  
Сервер автоматически определит наличие GPU:
- ✅ **RTX 4090 найдена**: Используется ускорение для обработки
- ⚠️ **GPU не найдена**: Работает на CPU

### Переменные окружения (опционально)
```bash
# Установите эти переменные для продакшена:
JWT_SECRET_KEY=your-production-jwt-secret-key
OPENROUTER_API_KEY=your-openrouter-api-key  
PERPLEXITY_API_KEY=your-perplexity-api-key
```

## 🧪 Тестирование

### 1. Тест здоровья сервера
```bash
curl http://localhost:5000/health
```
**Ожидается**: `{"status": "healthy", "message": "Server is running"}`

### 2. Тест генерации Device ID  
```bash
curl http://localhost:5000/auth/generate-device-id
```
**Ожидается**: `{"device_id": "device_abc123...", "instructions": "..."}`

### 3. Тест регистрации устройства
```bash
curl -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"device_id": "device_test123"}'
```
**Ожидается**: JWT токен в ответе

### 4. Полный цикл iOS аутентификации
```bash
# 1. Генерируем device_id
DEVICE_ID=$(curl -s http://localhost:5000/auth/generate-device-id | jq -r '.device_id')

# 2. Регистрируем устройство  
TOKEN=$(curl -s -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"device_id\": \"$DEVICE_ID\"}" | jq -r '.access_token')

# 3. Получаем информацию о пользователе
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/auth/me
```

## 🚨 Устранение проблем

### Сервер не запускается
- Проверьте, что порт 5000 свободен: `netstat -an | find ":5000"`
- Убедитесь что Python 3.8+ установлен: `python --version`
- Проверьте виртуальное окружение: `.venv\Scripts\activate`

### iOS App не подключается  
- Убедитесь что сервер доступен: `curl http://your-server-ip:5000/health`
- Проверьте настройки firewall на Windows
- Убедитесь что в iOS приложении указан правильный IP адрес сервера

### База данных ошибки
- Удалите файл `ai_conspects.db` для пересоздания базы
- Проверьте права доступа к директории

### GPU не определяется
- Установите PyTorch с CUDA поддержкой: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Проверьте драйверы NVIDIA: `nvidia-smi`

## 📊 Мониторинг

### Логи сервера
Сервер выводит логи в консоль:
- ✅ **INFO**: Нормальная работа
- ⚠️ **WARNING**: Предупреждения
- ❌ **ERROR**: Ошибки

### Health Check
```bash
# Проверка каждые 30 секунд
while true; do 
  curl -s http://localhost:5000/health | jq '.status'
  sleep 30
done
```

### База данных
SQLite база создается автоматически в файле `ai_conspects.db`:
- **users** - Пользователи
- **notes** - Заметки  
- **batches** - Батчи обработки аудио

## 🎯 Готово для продакшена

Сервер готов для продакшена с минимальными изменениями:
1. Поменяйте `JWT_SECRET_KEY` на надежный ключ
2. Настройте HTTPS (рекомендуется nginx reverse proxy)  
3. Мигрируйте на PostgreSQL для лучшей производительности
4. Настройте мониторинг и алерты
5. Добавьте backup базы данных

---

**🔥 Сервер готов к работе с вашим iOS приложением!**

Все endpoints соответствуют ожиданиям iOS приложения. Просто запустите `start_server.bat` и ваше приложение заработает.