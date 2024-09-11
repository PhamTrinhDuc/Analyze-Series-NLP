import scrapy
from configs.configuratior import CONFIGURATOR


class BlogSiper(scrapy.Spider):
    def __init__(self):
        self.URL = CONFIGURATOR.url_website
        self.NAME = CONFIGURATOR.name_data

    def parse(self, response):
        