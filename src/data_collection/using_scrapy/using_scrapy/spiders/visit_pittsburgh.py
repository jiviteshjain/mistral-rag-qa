import os
import sys
from typing import Iterable, Any

import scrapy

sys.path.append(os.path.join(os.path.abspath("../../")))
from util.util import *


class VisitPittsburghSpider(scrapy.spiders.SitemapSpider):
    name = "visit_pittsburgh"

    allowed_domains = ["visitpittsburgh.com"]

    sitemap_urls = [
        "https://www.visitpittsburgh.com/sitemaps-1-sitemap.xml",
    ]

    sitemap_rules = [
        ("/blog/", "parse_blog"),
        ("", "parse"),
    ]

    exclude_list = [
        "/pattern-library/",
        "/pattern-library-2/",
        "/test-page-one/",
        "/media-categories/",
        "sitemaps-1-categorygroup-mediaCategories-1-sitemap",
        "/willkommen/",
        "/de-welcome/",
        "/pittsburgh-international-airport-english/",
        "flypittsburgh.com",
        "/persona-quiz-categories/",
        "sitemaps-1-categorygroup-personaQuizCategories-1-sitemap",
    ]

    def sitemap_filter(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            if not any([x in str(entry.get("loc")).lower() for x in self.exclude_list]):
                yield entry

    def parse_blog(self, response: scrapy.http.Response):
        blog_title = process(
            response.css(
                "article.detail__inner div.page-title__content h1 ::text"
            ).getall()
        )

        blog_content = [
            # Adds the title, excerpt, and summary at the top.
            process(
                response.css(
                    "article.detail__inner div.page-title__content ::text"
                ).getall()
            ),
            # Adds the content.
            process(
                response.css(
                    "article.detail__inner div.text div.text__inner ::text"
                ).getall()
            ),
        ]

        yield {
            "url": response.url,
            "title": blog_title,
            "date_accessed": today(),
            "text_content": join(blog_content),
        }

    def parse(self, response: scrapy.http.Response):
        title = process(response.css("main h1 ::text").getall())

        content = process(
            response.xpath(
                "//body//*[not(ancestor-or-self::div[@class='header']) and not(ancestor-or-self::*[contains(@class, 'nav__link')]) and not(self::script) and not(ancestor-or-self::footer)]/text()"
            ).getall()
        )
        # content = process(
        #     response.xpath(
        #         f"//body//*[not(ancestor-or-self::div[{xpath_match_class('header')}]) and not(ancestor-or-self::*[{xpath_match_class('modal')}]) and not(self::script) and not(ancestor-or-self::footer)]/text()"
        #     ).getall()
        # )

        if len(content) > 0:
            yield {
                "url": response.url,
                "title": title,
                "date_accessed": today(),
                "text_content": content,
            }
