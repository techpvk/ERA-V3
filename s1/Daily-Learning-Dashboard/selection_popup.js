document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['tempSelection'], (result) => {
    if (result.tempSelection) {
      document.getElementById('selectedText').innerHTML = result.tempSelection.html;
    }
  });

  document.getElementById('saveSelection').addEventListener('click', () => {
    const description = document.getElementById('description').value;
    const tags = document.getElementById('tags').value.split(',').map(tag => tag.trim());
    
    chrome.storage.local.get(['tempSelection'], (result) => {
      if (result.tempSelection) {
        chrome.runtime.sendMessage({
          action: "saveSelection",
          content: result.tempSelection.text,
          html: result.tempSelection.html,
          url: result.tempSelection.url,
          description: description,
          tags: tags
        }, () => {
          console.log("Selection saved:", result.tempSelection.text);
          chrome.storage.local.remove('tempSelection');
          window.close();
        });
      }
    });
  });
});
