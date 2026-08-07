---
layout: page
title: Archive
description: "Archive of posts on finance, data, and research."
permalink: /sitearchive/
sitemap: false
header-img: "img/1316566.jpg"
Order: 2
---

<section id="archive">
  <section class="topic-archive" aria-labelledby="topics-title">
    <h2 id="topics-title">Browse by topic</h2>
    {% assign sorted_tags = site.tags | sort %}
    <ul class="topic-index">
      {% for tag in sorted_tags %}
        {% assign tag_name = tag[0] %}
        {% assign tag_posts = tag[1] %}
        <li>
          <a href="#topic-{{ tag_name | slugify }}">{{ tag_name }}</a>
          <span aria-label="{{ tag_posts | size }} posts">{{ tag_posts | size }}</span>
        </li>
      {% endfor %}
    </ul>

    <div class="topic-groups">
      {% for tag in sorted_tags %}
        {% assign tag_name = tag[0] %}
        {% assign tag_posts = tag[1] %}
        <section class="topic-group" id="topic-{{ tag_name | slugify }}">
          <h3>{{ tag_name }}</h3>
          <ul>
            {% for post in tag_posts %}
              <li><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a> <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y" }}</time></li>
            {% endfor %}
          </ul>
        </section>
      {% endfor %}
    </div>
  </section>

  <section class="year-archive" aria-labelledby="years-title">
  <h2 id="years-title">Browse by year</h2>
  {% assign current_year = '' %}
  {% for post in site.posts %}
      {% assign post_year = post.date | date: "%Y" %}
      {% if post_year != current_year %}
        {% unless forloop.first %}
        </ul>
        {% endunless %}
        <h3>{{ post_year }}</h3>
        <ul class="past">
        {% assign current_year = post_year %}
      {% endif %}
      <li><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: " %d %b " }}</time><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></li>
  {% endfor %}
  {% if site.posts != empty %}
  </ul>
  {% endif %}
  </section>
</section>
