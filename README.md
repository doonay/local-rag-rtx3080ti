# Local RAG для RTX 3080 Ti

Локальная RAG-система для загрузки PDF/TXT/MD, поиска по документам и генерации ответа со ссылками на найденные фрагменты.

## Запуск в один клик без Docker

Для обычного использования на этом Windows-компьютере Docker и WSL не нужны. Дважды щёлкните [start-local.bat](start-local.bat). При первом запуске скрипт автоматически:

- создаст изолированное Python-окружение `.venv-local`;
- установит PyTorch с CUDA 12.8 и остальные зависимости;
- скачает официальную переносимую Windows-сборку llama.cpp;
- скачает `Qwen3-8B-Q4_K_M.gguf`;
- запустит Qwen3 на GPU, локальное API и откроет браузер.

Первый запуск скачивает несколько гигабайт и может быть долгим. Последующие запуски не требуют установки и выполняются одним двойным кликом. Для остановки используйте [stop-local.bat](stop-local.bat). Логи находятся в `.local/logs`, индекс документов — в `.local/qdrant`.

Нативный режим использует встроенный Qdrant Local, поэтому не требует отдельного сервера базы данных. Для ручной предварительной установки можно запустить [setup-local.bat](setup-local.bat).

## Текущее состояние

Проект приведён из чернового в согласованное базовое состояние:

- API, обработчик документов, embedder и reranker импортируются как Python-пакеты;
- случайные «заглушечные» эмбеддинги удалены — недоступная модель теперь даёт честный `503`;
- Qdrant-коллекция с именованными dense+sparse векторами создаётся автоматически;
- BGE-M3 строит dense и learned sparse представления, Qdrant объединяет их server-side RRF;
- Qwen3-Reranker повторно оценивает top-20 кандидатов и передаёт LLM только top-5;
- документы дедуплицируются по SHA-256, их можно просматривать и удалять из Web UI;
- Dockerfile копируют все необходимые модули, версии базовых образов закреплены;
- unit-тесты не требуют поднятых моделей и отделены от integration/E2E;
- профиль Compose рассчитан на RTX 3080 Ti с 12 ГБ VRAM.

Пайплайн запроса:

```text
браузер → API → BGE-M3 dense+sparse → Qdrant RRF → Qwen3 Reranker 0.6B → Qwen3 8B AWQ → API
             CPU                                      GPU                  GPU
```

## Почему такой GPU-профиль

`Qwen/Qwen3-8B-AWQ` работает в non-thinking режиме и реалистично помещается на 3080 Ti. vLLM ограничен 70% VRAM (примерно 8.6 ГБ), оставшаяся память нужна `Qwen3-Reranker-0.6B` и CUDA-контексту. BGE-M3 работает на CPU, иначе три модели будут конкурировать за 12 ГБ и периодически падать с OOM. Подробное обоснование — в [docs/MODELS.md](docs/MODELS.md).

14B AWQ на этой карте возможна только с коротким контекстом и переносом reranker на CPU, но для стабильного постоянно работающего стека её использовать не стоит.

## Требования

- Windows 11 с WSL2 и Docker Desktop в режиме Linux containers;
- аппаратная виртуализация Intel VT-x/AMD SVM включена в BIOS/UEFI;
- актуальный NVIDIA-драйвер и включённая GPU-интеграция Docker/WSL;
- 32 ГБ RAM желательно, 24 ГБ — практический минимум;
- 50–80 ГБ свободного места на первый pull/build и кэши моделей.

Проверенная карта в этой системе: RTX 3080 Ti, 12 288 MiB, compute capability 8.6, драйвер 591.74.

Если Docker сообщает `HCS_E_HYPERV_NOT_INSTALLED`, выполните [настройку WSL2 и виртуализации](docs/WINDOWS_SETUP.md).

## Запуск через Docker — опционально

1. Запустите Docker Desktop и дождитесь состояния Engine running.
2. Создайте локальную конфигурацию:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Проверьте доступ Docker к видеокарте:

   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

4. Соберите и запустите стек:

   ```powershell
   docker compose pull
   docker compose build --pull
   docker compose up -d
   ```

5. Следите за первой загрузкой моделей (она может занять десятки минут):

   ```powershell
   docker compose logs -f embedder reranker vllm
   ```

6. Проверьте состояние и откройте интерфейс:

   ```powershell
   docker compose ps
   Invoke-RestMethod http://localhost:8000/health
   ```

   Web UI: <http://localhost:3000>, Swagger: <http://localhost:8000/docs>, Qdrant: <http://localhost:6333/dashboard>.

## Проверки разработчика

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
docker compose config --quiet
node --check web/app.js
```

Обычный `pytest` запускает только unit-тесты. Integration/E2E требуют поднятого стека и включаются явно, чтобы отсутствие локальных моделей не маскировалось случайными падениями.

При поднятом стеке они запускаются явно:

```powershell
$env:RUN_INTEGRATION = "1"
python -m pytest -m integration -o addopts="" -q
$env:RUN_E2E = "1"
python -m pytest -m e2e -o addopts="" -q
```

## Если не хватает VRAM

Остановите стек, в `.env` уменьшите `VLLM_GPU_MEMORY_UTILIZATION` до `0.68` и `VLLM_MAX_MODEL_LEN` до `4096`, затем снова выполните `docker compose up -d`. Не поднимайте embedder на CUDA одновременно с vLLM и reranker.

Если vLLM не стартует, сначала смотрите `docker compose logs vllm`, затем `nvidia-smi`. Типичные причины: Docker не видит GPU, модель не скачалась, остался другой процесс на GPU или слишком высок лимит VRAM.

## Данные и сброс индекса

Qdrant, модели, кэши и временные файлы хранятся внутри проекта в `.local`. В Docker-режиме это `.local/docker`, в нативном Windows-режиме — `.local/cache`, `.local/models` и `.local/tmp`. Старый индекс из версии до hybrid dense+sparse несовместим с новой схемой; его нужно один раз пересоздать. То же требуется после смены embedding-модели:

```powershell
docker compose down
Remove-Item -LiteralPath .local\docker\qdrant -Recurse -Force
docker compose up -d
```

Эта команда безвозвратно удаляет загруженный индекс документов. Кэш моделей остаётся.

## Лицензия и сторонние компоненты

Исходный код этого репозитория распространяется по лицензии [MIT](LICENSE).

Лицензия MIT относится только к коду репозитория и не изменяет условия использования сторонних моделей и инструментов. Веса моделей в репозиторий не входят и скачиваются из официальных источников. Перед использованием и распространением проверьте актуальные условия в карточках [Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ), [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [BGE-M3](https://huggingface.co/BAAI/bge-m3) и в репозитории [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Публикация и хостинг

GitHub используется для хранения исходного кода, документации и релизов. GitHub Pages обслуживает только статические файлы и не может запустить Python API, Qdrant или CUDA-модели этого проекта. Для полноценной работы систему нужно запускать локально либо разворачивать на отдельном сервере с подходящей видеокартой.

## Что делать дальше

Подробный порядок работ находится в [docs/ROADMAP.md](docs/ROADMAP.md). Самые важные следующие шаги: реальный GPU smoke-test, полноценные integration/E2E-тесты, удаление/дедупликация документов, настоящий dense+sparse hybrid search, OCR и измеримый набор RAG-evaluation.
