# Monthly Archive on Octupress
# rcmdnk (https://github.com/rcmdnk/monthly-archive)


module Jekyll
  class MonthlyArchive < Liquid::Tag
    def initialize(tag_name, markup, tokens)
      if markup.strip =~ /\s*month_format:(\w+)/i
        markup = markup.strip.sub(/month_format:\w+/i,'')
      end
      super
    end

    def render(context)
      config = context.registers[:site].config

      count = config['monthly_archive_count']
      open = config['monthly_archive_open']
      format = '%b'
      upcase = true
      case config['monthly_archive_format']
      when 'Jan'
        format = '%b'
        upcase = false
      when 'JAN'
        format = '%b'
        upcase = true
      when 'January'
        format = '%B'
        upcase = false
      when 'JANUARY'
        format = '%B'
        upcase = true
      when '01'
        format = '%m'
        upcase = false
      when '1'
        format = '%-m'
        upcase = false
      end

      if config['permalink'].start_with?("/blog/:year/")
        archive_dir = "/blog"
      else
        archive_dir = ""
      end
      html = ""
      html << "<div class=\"monthly_archive\">"
      html << "<ul>"
      posts = context.registers[:site].posts.reverse
      posts_years = posts.group_by{|c| {"year" => c.date.year}}
      posts_years.each_with_index do |(key_year, posts_year), index|
        html << "<li><span class=\"monthly_archive_year_span\">#{key_year["year"]}"
        html << " (#{posts_year.count})" if count
        html << "</span></li>"
        html << "<ul class=\"monthly_archive_month_list"
        if open == 'all' or\
          (open == 'first' and index == 0)
           html << " monthly_archive_open\">"
        else
           html << " monthly_archive_close\">"
        end
        posts_months = posts_year.group_by{|c| {\
          "month_format" => c.date.strftime(format),\
          "month" => c.date.strftime('%m')}}
        posts_months.each do |key_month, posts_month|
          month = key_month["month_format"]
          month = month.upcase if upcase
          html << "<li>"
          html << "<a href='#{archive_dir}/#{key_year["year"]}/#{key_month["month"]}/'>#{month}"
          html << " (#{posts_month.count})" if count
          html << "</a></li>"
        end
        html << "</ul>"
      end
      html << "</ul>"
      html << "</div>"
      html
    end
  end
end

Liquid::Template.register_tag('monthly_archive', Jekyll::MonthlyArchive)

