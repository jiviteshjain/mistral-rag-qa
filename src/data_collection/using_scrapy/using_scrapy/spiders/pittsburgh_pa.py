import os
import sys
from typing import Iterable, Any

import scrapy

sys.path.append(os.path.join(os.path.abspath("../../")))
from util.util import *


class PittsburghPaSpider(scrapy.spiders.SitemapSpider):
    name = "pittsburgh_pa"

    sitemap_urls = [
        "https://pittsburghpa.gov/sitemap.xml",
    ]

    sitemap_rules = [
        (".pdf", "parse_pdf"),
        ("", "parse"),
    ]

    exclude_list = [
        "pittsburghpa.gov/ps/",  # This is the only language code included in the sitemap.
    ]

    pdf_path = "/Users/jiviteshjain/Documents/CMU/Coursework/Sem-1/ANLP/Assignment-2/anlp-ass-2/data/pittsburgh_pa_pdf_2/"

    def sitemap_filter(self, entries: Iterable[dict[str, Any]]):
        for entry in entries:
            if not any([x in str(entry.get("loc")).lower() for x in self.exclude_list]):
                yield entry

    def parse_pdf(self, response: scrapy.http.Response):
        file_name = response.url.split("/")[-1]
        file_path = os.path.join(self.pdf_path, file_name)

        self.logger.info("Saving PDF to %s", file_path)

        os.makedirs(self.pdf_path, exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(response.body)

    def parse(self, response: scrapy.http.Response):
        title = process(response.css("section div.content-area h1 ::text").getall())

        content = process(
            response.css("section div.content-area")
            .xpath(".//*[not(self::script)]/text()")
            .getall()
        )

        if len(content) > 0:
            yield {
                "url": response.url,
                "title": title,
                "date_accessed": today(),
                "text_content": content,
            }
