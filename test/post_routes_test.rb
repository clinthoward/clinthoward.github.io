#!/usr/bin/env ruby

require "cgi"
require "fileutils"
require "jekyll"
require "jekyll-feed"
require "jekyll-paginate"
require "json"
require "shellwords"
require "tmpdir"
require "uri"

ROOT = File.expand_path("..", __dir__)
FIXTURE = File.join(ROOT, "test", "fixtures", "posts", "2026-08-07-clean-url-test.md")
LEGACY_URLS = File.readlines(File.join(ROOT, "test", "legacy_post_urls.txt"), chomp: true).reject(&:empty?).sort.freeze
TEST_URL = "/blog/clean-url-test/"
CANONICAL_URL = "https://clinthoward.github.io#{TEST_URL}"

def assert(condition, message)
  raise "FAIL: #{message}" unless condition
end

def copy_source(destination)
  FileUtils.mkdir_p(destination)
  tracked_files = `git -C #{Shellwords.escape(ROOT)} ls-files -z`.split("\0")

  tracked_files.each do |relative_path|
    next if relative_path.start_with?("test/")

    source = File.join(ROOT, relative_path)
    target = File.join(destination, relative_path)
    FileUtils.mkdir_p(File.dirname(target))
    FileUtils.cp(source, target)
  end

  FileUtils.cp_r(File.join(ROOT, "test"), File.join(destination, "test"))
end

def build_site(source, destination)
  config = Jekyll.configuration(
    "source" => source,
    "destination" => destination,
    "quiet" => true,
    "disable_disk_cache" => true
  )
  site = Jekyll::Site.new(config)
  site.process
  site
end

def assert_post_routes_declared(source)
  Dir.glob(File.join(source, "_posts", "*.*")).each do |post_path|
    front_matter = File.read(post_path).match(/\A---\s*\n(.*?)\n---\s*\n/m)&.captures&.first.to_s
    has_permalink = front_matter.match?(/^permalink:\s*\S+/)
    has_valid_slug = front_matter.match?(/^slug:\s*[a-z0-9]+(?:-[a-z0-9]+)*\s*$/)

    assert(has_permalink || has_valid_slug, "#{File.basename(post_path)} needs an explicit permalink or a lowercase, hyphenated slug")
  end
end

def output_path(destination, url)
  File.join(destination, url.delete_prefix("/"), "index.html")
end

def assert_contains(path, text, description)
  assert(File.file?(path), "missing #{description}: #{path}")
  assert(File.read(path).include?(text), "#{description} does not contain #{text.inspect}")
end

def assert_internal_links_resolve(destination)
  missing = []

  Dir.glob(File.join(destination, "**", "*.html")).each do |html_path|
    File.read(html_path).scan(/href=["']([^"']+)["']/).flatten.each do |raw_href|
      next unless raw_href.start_with?("/")
      next if raw_href.start_with?("//")

      href = CGI.unescapeHTML(raw_href).split(/[?#]/, 2).first
      next if href.nil? || href.empty?

      decoded = URI::DEFAULT_PARSER.unescape(href).delete_prefix("/")
      candidates = if href.end_with?("/")
                     [File.join(destination, decoded, "index.html")]
                   elsif File.extname(decoded).empty?
                     [File.join(destination, decoded), File.join(destination, decoded, "index.html")]
                   else
                     [File.join(destination, decoded)]
                   end

      missing << "#{html_path}: #{raw_href}" unless candidates.any? { |candidate| File.exist?(candidate) }
    rescue URI::InvalidURIError
      missing << "#{html_path}: invalid URL #{raw_href}"
    end
  end

  assert(missing.empty?, "unresolved internal links:\n#{missing.first(20).join("\n")}")
end

raise "Run this test through Bundler: bundle exec ruby test/post_routes_test.rb" unless defined?(Bundler)

Dir.mktmpdir("post-route-test-") do |temporary_directory|
  source = File.join(temporary_directory, "source")
  production_destination = File.join(temporary_directory, "production-site")
  fixture_destination = File.join(temporary_directory, "fixture-site")
  renamed_destination = File.join(temporary_directory, "renamed-fixture-site")

  copy_source(source)

  assert_post_routes_declared(source)
  production_site = build_site(source, production_destination)
  production_urls = production_site.posts.docs.map(&:url).sort
  assert(production_urls == LEGACY_URLS, "published post URLs changed:\n#{production_urls.join("\n")}")
  assert(LEGACY_URLS.length == 25, "legacy URL manifest must contain 25 posts")
  assert(LEGACY_URLS.uniq.length == LEGACY_URLS.length, "legacy URL manifest contains duplicates")
  LEGACY_URLS.each do |url|
    assert(File.file?(output_path(production_destination, url)), "missing legacy output #{url}")
  end
  production_text_files = Dir.glob(File.join(production_destination, "**", "*.{html,xml,json}"))
  assert(production_text_files.none? { |path| File.read(path).include?("clean-url-test") }, "fixture leaked into the production build")

  fixture_post = File.join(source, "_posts", File.basename(FIXTURE))
  FileUtils.cp(FIXTURE, fixture_post)
  assert_post_routes_declared(source)
  fixture_site = build_site(source, fixture_destination)
  fixture_urls = fixture_site.posts.docs.map(&:url).sort
  assert(fixture_urls == (LEGACY_URLS + [TEST_URL]).sort, "fixture build has unexpected post URLs")

  fixture_html = output_path(fixture_destination, TEST_URL)
  assert(File.file?(fixture_html), "clean URL output is missing")
  assert(!File.exist?(output_path(fixture_destination, "/testing/2026/08/07/clean-url-test/")), "category/date URL was generated")
  assert_contains(fixture_html, %(<link rel="canonical" href="#{CANONICAL_URL}">), "fixture canonical metadata")
  assert_contains(fixture_html, %(<meta property="og:url" content="#{CANONICAL_URL}">), "fixture Open Graph metadata")

  json_ld_blocks = File.read(fixture_html).scan(%r{<script type="application/ld\+json">\s*(.*?)\s*</script>}m).flatten.map { |block| JSON.parse(block) }
  blog_posting = json_ld_blocks.find { |block| block["@type"] == "BlogPosting" }
  assert(blog_posting, "fixture BlogPosting JSON-LD is missing")
  assert(blog_posting["url"] == CANONICAL_URL, "fixture JSON-LD URL is wrong")
  assert(blog_posting.dig("mainEntityOfPage", "@id") == CANONICAL_URL, "fixture JSON-LD page ID is wrong")

  assert_contains(File.join(fixture_destination, "feed.xml"), CANONICAL_URL, "fixture feed")
  assert_contains(File.join(fixture_destination, "sitemap.xml"), CANONICAL_URL, "fixture sitemap")
  archive_html = File.join(fixture_destination, "sitearchive", "index.html")
  assert_contains(archive_html, %(href="#{TEST_URL}"), "fixture archive link")
  assert_contains(archive_html, %(id="topic-test"), "fixture topic section")
  assert_contains(output_path(fixture_destination, "/portfolio/2023/12/10/PhDThesis/"), %(href="#{TEST_URL}"), "fixture related-post link")
  assert_internal_links_resolve(fixture_destination)

  renamed_fixture = File.read(fixture_post)
    .sub('title: "Clean URL Test Post"', 'title: "Renamed Clean URL Test Post"')
    .sub("category: Testing", "category: Essays")
  File.write(fixture_post, renamed_fixture)
  renamed_site = build_site(source, renamed_destination)
  renamed_urls = renamed_site.posts.docs.map(&:url)
  assert(renamed_urls.include?(TEST_URL), "title or category change altered the explicit slug URL")
  assert(!File.exist?(output_path(renamed_destination, "/essays/2026/08/07/clean-url-test/")), "renamed category affected the URL")

  puts "PASS: 25 legacy URLs preserved"
  puts "PASS: unpublished fixture generated only at #{TEST_URL}"
  puts "PASS: canonical, discovery, related-post, and internal links verified"
  puts "PASS: title and category changes left the fixture URL unchanged"
end
