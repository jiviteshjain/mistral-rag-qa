import os
import sys
from typing import Iterable, Any

import scrapy
import scrapy.http

sys.path.append(os.path.join(os.path.abspath("../../")))
from util.util import *


class SteelersSpider(scrapy.spiders.SitemapSpider):
    name = "steelers"

    allowed_domains = ["steelers.com"]

    sitemap_urls = [
        "https://www.steelers.com/sitemap.xml",
    ]

    sitemap_rules = [
        ("", "parse"),
    ]

    exclude_list = [
        "/manifest.json",
        "/forms/",
        "/test-page",
        "/video/",
        "/media/",
        "/audio/",
        "/legal/",
        "/game-day/2015",  # We want to include articles in game-day without the year.
        "/game-day/2016",
        "/game-day/2017",
        "/game-day/2018",
        "/game-day/2019",
        "/game-day/2020",
        "/game-day/2021",
        "/game-day/2022",
        "/game-day/2023",
        "/game-day/2024",
        "/game-day/2025",
        "/game-day/2026",
        "/news/",
    ]

    def sitemap_filter(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            if not any([x in str(entry.get("loc")).lower() for x in self.exclude_list]):
                yield entry

    def parse(self, response: scrapy.http.Response):
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
