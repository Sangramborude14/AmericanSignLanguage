/**
 * Chrome Extension Background Service Worker
 */

chrome.runtime.onInstalled.addListener(() => {
    console.log('ASL Sign Language Translator Extension Installed.');
});

// Listen for messages from popup or sidepanel
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'open_side_panel') {
        chrome.windows.getCurrent((window) => {
            if (chrome.sidePanel && chrome.sidePanel.open) {
                chrome.sidePanel.open({ windowId: window.id });
            }
        });
        sendResponse({ status: 'ok' });
    }
    return true;
});
