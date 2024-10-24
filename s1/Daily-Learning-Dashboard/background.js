chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "saveLearning",
    title: "Save to Learning Dashboard",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "saveLearning") {
    chrome.tabs.sendMessage(tab.id, {action: "getSelection"}, response => {
      if (response && response.html) {
        chrome.windows.create({
          url: chrome.runtime.getURL("selection_popup.html"),
          type: "popup",
          width: 400,
          height: 400
        }, (window) => {
          chrome.storage.local.set({ 
            tempSelection: {
              html: response.html,
              text: response.text,
              url: response.url  // Use the URL from the content script
            }
          });
        });
      }
    });
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "saveWebpage") {
    saveItem("webpage", request.url, request.description, null, null, request.tags);
  } else if (request.action === "saveSelection") {
    saveItem("selection", request.url, request.description, request.content, request.html, request.tags);
  }
});

function saveItem(type, url, description, content = null, html = null, tags = []) {
  chrome.storage.local.get({ learnings: [] }, (result) => {
    const learnings = result.learnings;
    learnings.push({
      type: type,
      url: url,
      description: description,
      content: content,
      html: html,
      tags: tags,
      timestamp: new Date().toISOString()
    });
    chrome.storage.local.set({ learnings: learnings }, () => {
      console.log("Item saved:", {type, url, description, content, html, tags});
    });
  });
}
