import os
import sys
from typing import Iterable, Any
import logging

import scrapy
import scrapy.http

sys.path.append(os.path.join(os.path.abspath("../../")))
from util.util import *


class PiratesSpider(scrapy.spiders.SitemapSpider):
    name = "pirates"

    allowed_domains = ["mlb.com"]

    sitemap_urls = [
        "https://www.mlb.com/sitemaps/pages/en/pirates.xml.gz",
        "https://www.mlb.com/sitemaps/48-hr-news.xml.gz",
        "https://www.mlb.com/sitemaps/weekly-news/index.xml.gz",
    ]

    sitemap_rules = [
        ("/pirates/", "parse_pirates"),  # Everything under /pirates/ is accepted.
        # ("/news/", "parse_news"),
        # ("/press-release/", "parse_news"),
        ("", "parse"),
    ]

    exclude_list = [
        "/es/",
        "/weekly-video/",
        "/video/",
        "/image/",
        "/hall-of-fame/",  # These pages have text as images, so can't be scraped.
    ]

    news_include_list = [
        "pirates",
        "pirate",
        "bucs",
        "buccos",
        "buc",
        "bucco",
        "pitt",
        "pittsburgh",
        "pnc",
        "allegheny",
        "forbes",  # forbes field
        "three-rivers",  # three rivers stadium
    ]

    def sitemap_filter(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            if "sitemap" in str(entry.get("loc")).lower():
                yield entry
            elif any([x in str(entry.get("loc")).lower() for x in self.exclude_list]):
                continue
            elif "/pirates/" in str(entry.get("loc")):
                yield entry

    def parse(self, response: scrapy.http.Response):
        return

    def parse_pirates(self, response: scrapy.http.Response):
        title = process(response.css("main h1 ::text").getall())

        content = process(
            response.xpath(
                f"//main//*[not(ancestor-or-self::div[{xpath_match_class('p-related-links')}]) and not(ancestor-or-self::script) and not(ancestor-or-self::style) and not(ancestor-or-self::iframe)]/text()"
            ).getall()
        )

        if len(content) > 0:
            yield {
                "url": response.url,
                "title": title,
                "date_accessed": today(),
                "text_content": content,
            }

    def parse_news(self, response: scrapy.http.Response):
        if not any([x in str(response.url).lower() for x in self.news_include_list]):
            return

        title = process(response.css("main h1 ::text").getall())
        content = process(
            response.xpath(
                f"//article//*[ancestor-or-self::div[{xpath_match_class('story-part')} and {xpath_match_class('markdown')}] and not(ancestor-or-self::script) and not(ancestor-or-self::script) and not(ancestor-or-self::style) and not(ancestor-or-self::iframe)]/text()"
            ).getall()
        )

        maybe_author = response.css("article header div.contributor__text ::text").get()

        if maybe_author is None:
            maybe_author = response.css(
                "article div.article--legacy__byline-name ::text"
            ).get()

        if maybe_author is None:
            maybe_author = ""
            logging.warning(f"Author not found for {response.url}")

        author = process(maybe_author)

        maybe_news_date = response.css("article header div ::text").get()

        if maybe_news_date is None:
            maybe_news_date = response.css(
                "article time.article--legacy__byline-date ::text"
            ).get()

        if maybe_news_date is None:
            maybe_news_date = ""
            logging.warning(f"Date not found for {response.url}")

        news_date = parse_date(process(maybe_news_date))

        other_dates = find_dates(content, news_date)

        if len(content) > 0:
            yield {
                "url": response.url,
                "title": title,
                "date_accessed": today(),
                "text_content": join([news_date, author, content]),
                "associated_dates": [news_date, *other_dates],
            }
