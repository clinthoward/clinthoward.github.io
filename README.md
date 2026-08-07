
## What?
The repo behind my [blog](clinthoward.github.io). Feel free to use anything located here.

#Clean Blog by Start Bootstrap - Jekyll Version

The official Jekyll version of the Clean Blog theme by [Start Bootstrap](http://startbootstrap.com/).

###[View Live Demo &rarr;](http://blackrockdigital.github.io/startbootstrap-clean-blog-jekyll/)

## Publishing posts

New posts use `/blog/<slug>/`. Add a short, lowercase, hyphenated `slug` to
the front matter of every new post:

```yaml
---
layout: post
title: "Juggling a full-time career with a PhD"
slug: juggling-work-and-a-phd
category: Productivity
tags: [study, thesis, career, phd]
---
```

Do not change the slug after publishing. Titles, categories, and tags may
change without changing the URL. Existing posts declare their old URLs with
an explicit `permalink`.

Run the route test before publishing:

```sh
bundle exec ruby test/post_routes_test.rb
```
