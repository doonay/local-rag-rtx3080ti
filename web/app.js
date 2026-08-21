// Конфигурация
const API_URL = '/api/v1';
const MAX_MESSAGES = 50;

// State
let messages = [];
let isProcessing = false;
let uploadedDocs = new Map();

// DOM элементы
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const docList = document.getElementById('docList');
const clearChatBtn = document.getElementById('clearChat');

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
    loadFromStorage();
    updateUI();
    setupEventListeners();
    loadDocuments();
});

// Настройка событий
function setupEventListeners() {
    // Отправка вопроса
    sendBtn.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // Загрузка файлов
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload();
        }
    });

    // Очистка чата
    clearChatBtn.addEventListener('click', clearChat);
}

// Отправка вопроса
async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question || isProcessing) return;

    // Добавляем сообщение пользователя
    addMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';
    
    // Показываем индикатор загрузки
    isProcessing = true;
    updateUI();
    
    try {
        const response = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: 3,
                search_limit: 10,
                temperature: 0.7,
                max_tokens: 512
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка сервера');
        }

        const data = await response.json();
        
        // Добавляем ответ ассистента
        addMessage('assistant', data.answer, data.sources);
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('assistant', `❌ Ошибка: ${error.message}`);
    } finally {
        isProcessing = false;
        updateUI();
        saveToStorage();
    }
}

// Добавление сообщения
function addMessage(role, content, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    
    // Форматируем текст с переносами
    const text = document.createElement('p');
    text.textContent = content;
    contentDiv.appendChild(text);
    
    // Добавляем источники
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        const heading = document.createElement('strong');
        heading.textContent = '📎 Источники:';
        sourcesDiv.appendChild(heading);
        
        sources.forEach((source, index) => {
            const item = document.createElement('div');
            item.className = 'source-item';
            const score = (source.score * 100).toFixed(1);
            const filename = document.createElement('span');
            filename.textContent = `${index + 1}. ${source.filename}`;
            const scoreBadge = document.createElement('span');
            scoreBadge.className = 'score';
            scoreBadge.textContent = `${score}%`;
            item.appendChild(filename);
            item.appendChild(scoreBadge);
            sourcesDiv.appendChild(item);
        });
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Прокрутка вниз
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Загрузка файлов
async function handleFileUpload() {
    const files = fileInput.files;
    if (!files.length) return;
    
    uploadStatus.innerHTML = '<span class="info">⏳ Загрузка...</span>';
    
    let success = 0;
    let errors = 0;
    const errorMessages = [];
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                success++;
                uploadedDocs.set(data.document_id, data.filename);
            } else {
                errors++;
                const error = await response.json().catch(() => ({}));
                const detail = typeof error.detail === 'string'
                    ? error.detail
                    : (error.detail?.message || response.statusText || `HTTP ${response.status}`);
                errorMessages.push(`${file.name}: ${detail}`);
                console.error(`Upload failed for ${file.name}:`, detail);
            }
        } catch (error) {
            errors++;
            errorMessages.push(`${file.name}: ${error.message || 'ошибка сети'}`);
            console.error(`Error uploading ${file.name}:`, error);
        }
    }
    
    // Обновляем статус
    if (success > 0) {
        uploadStatus.innerHTML = `<span class="success">✅ Загружено: ${success} файлов</span>`;
        updateDocList();
        // Включаем чат
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.placeholder = 'Введите ваш вопрос...';
    } else {
        uploadStatus.innerHTML = '';
        const errorStatus = document.createElement('span');
        errorStatus.className = 'error';
        errorStatus.textContent = `❌ ${errorMessages.join('; ') || 'Ошибка загрузки'}`;
        uploadStatus.appendChild(errorStatus);
    }
    
    if (errors > 0) {
        uploadStatus.innerHTML += ` <span class="error">(${errors} ошибок)</span>`;
    }
    
    fileInput.value = '';
}

// Обновление списка документов
function updateDocList() {
    if (uploadedDocs.size === 0) {
        docList.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">Нет загруженных документов</div>';
        return;
    }
    
    docList.innerHTML = '';
    uploadedDocs.forEach((name, documentId) => {
        const item = document.createElement('div');
        item.className = 'doc-item';
        const filename = document.createElement('span');
        filename.className = 'name';
        filename.textContent = `📄 ${name}`;
        const removeButton = document.createElement('button');
        removeButton.className = 'doc-delete';
        removeButton.type = 'button';
        removeButton.textContent = '×';
        removeButton.title = 'Удалить документ из индекса';
        removeButton.addEventListener('click', () => deleteDocument(documentId, name));
        item.appendChild(filename);
        item.appendChild(removeButton);
        docList.appendChild(item);
    });
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_URL}/documents`);
        if (!response.ok) return;
        const data = await response.json();
        uploadedDocs = new Map(
            data.documents.map(document => [document.document_id, document.filename])
        );
        updateDocList();
        updateUI();
    } catch (error) {
        console.warn('Could not load indexed documents', error);
    }
}

async function deleteDocument(documentId, filename) {
    if (!window.confirm(`Удалить «${filename}» из индекса?`)) return;
    try {
        const response = await fetch(`${API_URL}/documents/${documentId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Сервер не подтвердил удаление');
        uploadedDocs.delete(documentId);
        updateDocList();
        updateUI();
    } catch (error) {
        addMessage('assistant', `❌ Не удалось удалить документ: ${error.message}`);
    }
}

// Очистка чата
function clearChat() {
    const messages = chatMessages.querySelectorAll('.message');
    if (messages.length <= 1) return;
    
    // Оставляем только первое приветственное сообщение
    chatMessages.innerHTML = `
        <div class="message assistant">
            <div class="avatar">🤖</div>
            <div class="content">
                <p>Чат очищен. Задайте новый вопрос!</p>
            </div>
        </div>
    `;
    
    localStorage.removeItem('rag_chat_history');
}

// Обновление UI
function updateUI() {
    const hasDocs = uploadedDocs.size > 0;
    questionInput.disabled = isProcessing || !hasDocs;
    sendBtn.disabled = isProcessing || !hasDocs || !questionInput.value.trim();
    
    if (!hasDocs && !isProcessing) {
        questionInput.placeholder = 'Сначала загрузите документы...';
    } else if (isProcessing) {
        questionInput.placeholder = '⏳ Генерация ответа...';
        sendBtn.textContent = '⏳';
    } else {
        questionInput.placeholder = 'Введите ваш вопрос...';
        sendBtn.textContent = 'Отправить';
    }
}

// Сохранение в localStorage
function saveToStorage() {
    const messagesData = [];
    const messageElements = chatMessages.querySelectorAll('.message');
    messageElements.forEach(el => {
        const role = el.classList.contains('user') ? 'user' : 'assistant';
        const content = el.querySelector('.content p')?.textContent || '';
        messagesData.push({ role, content });
    });
    localStorage.setItem('rag_chat_history', JSON.stringify(messagesData));
}

// Загрузка из localStorage
function loadFromStorage() {
    const saved = localStorage.getItem('rag_chat_history');
    if (saved) {
        try {
            const data = JSON.parse(saved);
            // Очищаем чат (кроме приветствия)
            chatMessages.innerHTML = '';
            data.forEach(msg => {
                addMessage(msg.role, msg.content);
            });
        } catch (e) {
            console.warn('Failed to load chat history');
        }
    }
    
    // Загружаем список документов из localStorage
}

// Авто-высота textarea
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = questionInput.scrollHeight + 'px';
    updateUI();
});

// Функция для проверки статуса сервера
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            document.querySelector('.status-indicator')?.remove();
            uploadStatus.innerHTML = '<span class="success">🟢 Сервер подключен</span>';
        }
    } catch (e) {
        uploadStatus.innerHTML = '<span class="error">🔴 Сервер недоступен</span>';
    }
}

// Проверяем статус при загрузке
setTimeout(checkServerStatus, 1000);
