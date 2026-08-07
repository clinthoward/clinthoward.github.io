(function () {
  'use strict';

  var content = document.querySelector('.post-content');
  var toc = document.querySelector('.post-toc');
  var list = toc && toc.querySelector('.post-toc-list');

  if (!content || !toc || !list) {
    return;
  }

  var postTitle = (content.getAttribute('data-post-title') || '').trim().toLowerCase();
  var headings = Array.prototype.slice.call(content.querySelectorAll('h2, h3'));
  var removedRepeatedTitle = false;

  headings = headings.filter(function (heading) {
    var text = heading.textContent.trim();

    if (!text) {
      return false;
    }

    if (!removedRepeatedTitle && text.toLowerCase() === postTitle) {
      removedRepeatedTitle = true;
      return false;
    }

    return true;
  });

  if (headings.length < 3) {
    return;
  }

  function slugify(value) {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-') || 'section';
  }

  function uniqueId(heading) {
    var base = heading.id || slugify(heading.textContent);
    var id = base;
    var suffix = 2;
    var match = document.getElementById(id);

    while (match && match !== heading) {
      id = base + '-' + suffix;
      suffix += 1;
      match = document.getElementById(id);
    }

    heading.id = id;
    return id;
  }

  var currentSection = null;

  headings.forEach(function (heading) {
    var item = document.createElement('li');
    var link = document.createElement('a');

    link.href = '#' + uniqueId(heading);
    link.textContent = heading.textContent.trim();
    item.appendChild(link);

    if (heading.tagName === 'H3' && currentSection) {
      var nested = currentSection.querySelector('ol');

      if (!nested) {
        nested = document.createElement('ol');
        currentSection.appendChild(nested);
      }

      nested.appendChild(item);
    } else {
      list.appendChild(item);
      currentSection = heading.tagName === 'H2' ? item : null;
    }
  });

  toc.removeAttribute('hidden');
}());
