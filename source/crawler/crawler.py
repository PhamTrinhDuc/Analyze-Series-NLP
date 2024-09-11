import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    """
    Spider để crawl thông tin về jutsu từ trang web Naruto Fandom.

    Class này kế thừa từ scrapy.Spider và được thiết kế để trích xuất
    thông tin chi tiết về các jutsu (kỹ năng) từ trang web Naruto Fandom.

    Methods:
        parse(response): Phương thức khởi đầu để phân tích trang chính và 
                         tạo các request cho từng trang jutsu cụ thể.
        parse_article(response): Phân tích và trích xuất thông tin từ 
                                 trang chi tiết của mỗi jutsu.

    Usage:
        Spider này được sử dụng với Scrapy framework. 
        Để chạy file:
            scrapy runspider source/crawler/crawler.py -o data/jutsu.json
    """

    name = 'narutospider'
    start_urls = ["https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu"]

    def parse(self, response):
        """
        Phân tích trang chính và tạo request cho các trang con

        Args:
            response: Đối tượng response từ Scrapy chứa nội dung của trang chính.
        Yields:
            scrapy.Request: Các request cho từng trang con cụ thể.
        """
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                           callback=self.parse_jutsu)
            yield extracted_data

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)

    
    def parse_jutsu(self, response):
        
        """
        Phân tích và trích xuất thông tin từ trang chi tiết từ trang chính.

        Args:
            response: Đối tượng response từ Scrapy chứa nội dung của trang jutsu.

        Returns:
            dict: Một dictionary chứa thông tin về jutsu, bao gồm:
                - jutsu_name: Tên của jutsu
                - jutsu_type: Loại của jutsu (nếu có)
                - jutsu_description: Mô tả chi tiết về jutsu
        """

        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0]
        jutsu_name = jutsu_name.strip()

        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        jutsu_type=""
        if soup.find('aside'):
            aside = soup.find('aside')

            for cell in aside.find_all('div',{'class':'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip()

        soup.find('aside').decompose()

        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict (
            jutsu_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )