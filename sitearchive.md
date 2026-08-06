---
layout: page
title: Archive
description: "Archive of posts by Clint Howard on finance, data, and research."
permalink: /sitearchive/
sitemap: false
header-img: "img/1316566.jpg"
Order: 2
---

<section id="archive">
  <h2>This year's posts</h2>
  {% capture current_year %}{% cycle 0, site.posts.first.date | date: "%Y" %}{% endcapture %}
  <ul class="this">
    {% for post in site.posts %}
      {% capture post_year %}{{post.date | date: "%Y"}}{% endcapture %}
      {% if post_year != current_year %}
        </ul>
        <h2>{{ post.date | date: "%Y" }}</h2>
        <ul class="past">
        {% capture current_year %}{{post_year}}{% endcapture %}
      {% endif %}
      <li><time>{{ post.date | date: " %d %b " }}</time><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>
