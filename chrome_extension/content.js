/**
 * ASL Extension Content Script
 * Injects translated sign language text into active web inputs/editors
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'insert_text') {
        const success = insertTextIntoActiveElement(request.text);
        sendResponse({ success: success });
    }
    return true;
});

function insertTextIntoActiveElement(text) {
    if (!text) return false;

    const el = document.activeElement;
    if (!el) return false;

    // 1. Standard Input or Textarea
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        const start = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? el.value.length;
        const val = el.value;

        el.value = val.substring(0, start) + text + val.substring(end);
        el.selectionStart = el.selectionEnd = start + text.length;

        // Trigger input and change events for reactive frameworks (React, Vue, Angular)
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
    }

    // 2. Contenteditable div (Gmail, Slack, Discord, Google Docs, Notion, etc.)
    if (el.isContentEditable) {
        // Try execCommand first for full undo/redo stack compatibility
        const executed = document.execCommand('insertText', false, text);
        if (!executed) {
            const sel = window.getSelection();
            if (sel && sel.rangeCount > 0) {
                const range = sel.getRangeAt(0);
                range.deleteContents();
                const textNode = document.createTextNode(text);
                range.insertNode(textNode);
                range.setStartAfter(textNode);
                range.setEndAfter(textNode);
                sel.removeAllRanges();
                sel.addRange(range);
            } else {
                el.innerText += text;
            }
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        return true;
    }

    return false;
}
