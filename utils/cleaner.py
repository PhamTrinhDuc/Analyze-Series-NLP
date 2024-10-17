from bs4 import BeautifulSoup

class Cleaner:
    def __init__(self):
        pass

    def put_line_breaks(self, text: str) -> str:
        return text.replace("<\p>", "<\p>\n")
    
    def clean_html(self, html: str) -> str:
        cleaned_html = BeautifulSoup(html, "lxml").text
        return cleaned_html
    
    def clean(self, text: str) -> str:
        text = self.put_line_breaks(text)
        text = self.clean_html(text)
        return text.strip()