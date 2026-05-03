const phoneNumber = "ui_testing_user_" + Math.floor(Math.random() * 1000); 

let isRecording = false;
let mediaRecorder;
let audioChunks = [];

document.addEventListener("DOMContentLoaded", () => {
    // Trigger backend-driven first prompt (language selection flow)
    sendApiRequest({ phone_number: phoneNumber, message: "start", input_type: "text" });
});

function handleKeyPress(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
}

function sendQuickMessage(text) {
    appendUserMessage(text);
    sendApiRequest({ phone_number: phoneNumber, message: text, input_type: "text" });
}

function sendMessage() {
    const inputEl = document.getElementById('user-input');
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';
    appendUserMessage(text);
    sendApiRequest({ phone_number: phoneNumber, message: text, input_type: "text" });
}

function appendUserMessage(text) {
    const messages = document.getElementById("messages");
    const msgWrapper = document.createElement("div");
    msgWrapper.className = "message-wrapper user";
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    msgWrapper.innerHTML = `
        <div class="message">
            ${text}
            <div class="msg-footer">
                <span class="timestamp">${time}</span>
                <span class="material-symbols-outlined status-ticks">done_all</span>
            </div>
        </div>
    `;
    
    messages.appendChild(msgWrapper);
    scrollToBottom();
}

function appendSystemMessage(text) {
    const messages = document.getElementById("messages");
    const loader = document.createElement("div");
    loader.className = "typing";
    loader.id = "loader";
    loader.innerText = text;
    messages.appendChild(loader);
    scrollToBottom();
}

function removeLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.remove();
}

// Toggle scheme detail expand/collapse
function toggleSchemeDetail(detailId, cardId) {
    const detail = document.getElementById(detailId);
    const icon   = document.getElementById("icon-" + detailId);
    if (!detail) return;
    const isOpen = detail.style.display !== "none";
    detail.style.display = isOpen ? "none" : "block";
    if (icon) icon.textContent = isOpen ? "▼" : "▲";
    const card = document.getElementById(cardId);
    if (card) card.classList.toggle("expanded", !isOpen);
}

function appendBotMessage(data) {
    removeLoader();
    
    const messages = document.getElementById("messages");
    const msgWrapper = document.createElement("div");
    msgWrapper.className = "message-wrapper bot";
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // --- CHAT BUBBLE: short response text only ---
    const responseText = (data.response || "").replace(/\n/g, "<br>");
    let innerHtml = `<div class="message">${responseText}
        <div class="msg-footer">
            <span class="timestamp">${time}</span>
        </div>
    </div>`;

    // --- SCHEME CARDS: rendered OUTSIDE the chat bubble ---
    if (data.schemes && data.schemes.length > 0) {
        data.schemes.forEach((s, idx) => {
            const ts       = Date.now();
            const detailId = `sd-${ts}-${idx}`;
            const cardId   = `sc-${ts}-${idx}`;

            const docs  = Array.isArray(s.documents_required) ? s.documents_required :
                          (s.documents_required ? [String(s.documents_required)] : []);
            const steps = Array.isArray(s.application_process) ? s.application_process : [];
            const why   = Array.isArray(s.why_match) ? s.why_match :
                          (s.why_match ? [String(s.why_match)] : []);

            const docsHtml  = docs.length  ? `<ul class="scheme-list">${docs.map(d => `<li>${d}</li>`).join("")}</ul>`    : `<p class="scheme-na">Not listed in dataset</p>`;
            const stepsHtml = steps.length ? `<ol class="scheme-steps">${steps.map(st => `<li>${st}</li>`).join("")}</ol>` : `<p class="scheme-na">Visit official portal or nearest government office</p>`;
            const whyHtml   = why.length   ? `<ul class="scheme-list">${why.map(r => `<li>${r}</li>`).join("")}</ul>`      : "";

            const stateLabel = (s.state && s.state.trim()) ? s.state.trim() : "All India";
            const applyLink  = (s.application_link && s.application_link.trim()) ? s.application_link.trim() : "";

            innerHtml += `
            <div class="scheme-card" id="${cardId}">
                <div class="scheme-card-header" onclick="toggleSchemeDetail('${detailId}','${cardId}')">
                    <div class="scheme-title-row">
                        <span class="scheme-badge">${stateLabel}</span>
                        <span class="scheme-toggle-icon" id="icon-${detailId}">▼</span>
                    </div>
                    <h4 class="scheme-name">${s.scheme_name || "Scheme"}</h4>
                    <p class="scheme-summary">${s.benefits_summary || s.description || ""}</p>
                </div>
                <div class="scheme-detail" id="${detailId}" style="display:none;">
                    <div class="scheme-section">
                        <span class="scheme-section-title">📋 Description</span>
                        <p>${s.description || s.benefits_summary || "Not available"}</p>
                    </div>
                    <div class="scheme-section">
                        <span class="scheme-section-title">✅ Eligibility</span>
                        <p>${s.eligibility_text || "Check official portal for eligibility details"}</p>
                    </div>
                    <div class="scheme-section">
                        <span class="scheme-section-title">📄 Documents Required</span>
                        ${docsHtml}
                    </div>
                    <div class="scheme-section">
                        <span class="scheme-section-title">🪜 How to Apply</span>
                        ${stepsHtml}
                    </div>
                    ${whyHtml ? `<div class="scheme-section"><span class="scheme-section-title">🎯 Why Matched</span>${whyHtml}</div>` : ""}
                    <div class="scheme-card-actions">
                        ${applyLink ? `<a class="apply-link" href="${applyLink}" target="_blank" rel="noopener noreferrer">🌐 Official Portal</a>` : ""}
                        <button class="apply-btn" onclick="sendQuickMessage('apply for ${s.scheme_name}')">Apply Now →</button>
                    </div>
                </div>
            </div>`;
        });
    }
    
    if (data.audio_url) {
        innerHtml += `
        <audio controls autoplay style="margin-top: 10px; width: 100%;">
            <source src="${data.audio_url}" type="audio/mpeg">
        </audio>`;
    }
    
    msgWrapper.innerHTML = innerHtml;
    messages.appendChild(msgWrapper);
    scrollToBottom();
}

function scrollToBottom() {
    const container = document.getElementById("messages");
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
}

async function sendApiRequest(payload) {
    appendSystemMessage("Bot is typing...");
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        appendBotMessage(data);
    } catch (e) {
        removeLoader();
        appendBotMessage({ response: "Network Error: Could not connect to backend." });
    }
}

async function sendVoiceApiRequest(audioBlob) {
    appendSystemMessage("Processing voice...");
    
    try {
        const formData = new FormData();
        formData.append("phone_number", phoneNumber);
        formData.append("input_type", "voice");
        
        formData.append("audio", audioBlob, "audio_clip.wav");
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        appendBotMessage(data);
    } catch (e) {
        removeLoader();
        appendBotMessage({ response: "Voice upload failed." });
    }
}

// MediaRecorder Logic
async function toggleRecording() {
    const micBtn = document.getElementById("mic-btn");
    
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                sendVoiceApiRequest(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            audioChunks = [];
            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add("active");
            
            setTimeout(() => {
                if(isRecording) toggleRecording();
            }, 7000);
            
        } catch (e) {
            console.error("Mic Access Error:", e);
            alert("Microphone access denied. Please allow it to use Voice feature.");
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove("active");
    }
}
