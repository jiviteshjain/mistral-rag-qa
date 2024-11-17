import os
import sys
from typing import Iterable, Any
import logging

import scrapy
import scrapy.http

sys.path.append(os.path.join(os.path.abspath("../../")))
from util.util import *


class PenguinsSpider(scrapy.spiders.SitemapSpider):
    name = "penguins"

    allowed_domains = ["nhl.com"]

    sitemap_urls = [
        "https://www.nhl.com/sitemap/sitemap.xml",
        "https://www.nhl.com/sitemap/sitemap-stories.xml",
    ]

    sitemap_rules = [
        ("/news/", "parse_news"),  # Even /penguins/news/ will go here.
        ("/penguins/", "parse_penguins"),
        ("", "parse"),
    ]

    exclude_list = [
        "/video/",
        "/multimedia/",
    ]

    news_include_list = [
        "penguins",
        "penguin",
        "pens",
        "pen",
        "pitt",
        "pittsburgh",
        "allegheny",
        "ppg",
        "paints",
        "ppg-paints",
        "hunt",
        "hunt-armory",
    ]

    def sitemap_filter(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            if "sitemap" in str(entry.get("loc")).lower():
                yield entry
            elif any([x in str(entry.get("loc")).lower() for x in self.exclude_list]):
                continue
            elif "/penguins/" in str(entry.get("loc")):
                yield entry
            elif any([x in str(entry.get("loc")).lower() for x in ["/news/"]]) and any(
                [x in str(entry.get("loc")).lower() for x in self.news_include_list]
            ):
                yield entry

    def parse(self, response: scrapy.http.Response):
        return

    def parse_penguins(self, response: scrapy.http.Response):
        title = process(response.css("h1 ::text").getall())

        content = process(
            response.xpath(
                "//main//*[not(ancestor-or-self::script) and not(ancestor-or-self::style) and not(ancestor-or-self::iframe)]/text()"
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

        title = process(response.css("h1 ::text").getall())
        content = process(
            response.xpath(
                "//main//*[not(ancestor-or-self::script) and not(ancestor-or-self::style) and not(ancestor-or-self::iframe)]/text()"
            ).getall()
        )

        maybe_news_date = response.css("time ::text").get()
        if maybe_news_date is None:
            maybe_news_date = ""
            logging.warning(f"Date not found for {response.url}")

        news_date = parse_date(process(maybe_news_date))
        logging.info(f"Processed date: {maybe_news_date} into {news_date}")

        if len(content) > 0:
            yield {
                "url": response.url,
                "title": title,
                "date_accessed": today(),
                "text_content": join([news_date, maybe_news_date, content]),
                "associated_dates": [news_date],
            }
