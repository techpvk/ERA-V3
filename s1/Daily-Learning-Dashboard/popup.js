document.addEventListener('DOMContentLoaded', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    document.getElementById('url').value = tabs[0].url;
  });

  document.getElementById('saveWebpage').addEventListener('click', () => {
    const url = document.getElementById('url').value;
    const description = document.getElementById('description').value;
    const tags = document.getElementById('tags').value.split(',').map(tag => tag.trim());
    
    chrome.runtime.sendMessage({
      action: "saveWebpage",
      url: url,
      description: description,
      tags: tags
    });

    window.close();
  });

  document.getElementById('openDashboard').addEventListener('click', () => {
    chrome.tabs.create({ url: 'dashboard.html' });
  });
});
