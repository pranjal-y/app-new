import scrapy
import csv

class CredihealthSpider(scrapy.Spider):
    name = 'credihealth_spider'

    def start_requests(self):
        csv_path = '/Users/pranjalyadav/Desktop/credihealth.csv'
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                url = row[0]
                procedure_name = row[1]
                yield scrapy.Request(url=url, callback=self.parse, meta={'procedure_name': procedure_name})

    def parse(self, response):
        procedure_name = response.meta.get('procedure_name')
        procedure_price = response.xpath("//p[@class='color-green fs-28 fw-500 margin-b0']/text()").get()
        cred_procedure_price = response.xpath("//p[@class='color-green fs-28 fw-500 margin-b0']/text()").get()

        yield {
            'Name_of_disease': procedure_name,
            'url': response.url,
            'procedure_price': procedure_price.strip() if procedure_price else 'NA',
            'cred_procedure_price': cred_procedure_price.strip() if cred_procedure_price else 'NA',
        }

def run_spider():
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0',
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'procedure_prices.csv',
        'LOG_LEVEL': 'INFO',
        'DOWNLOAD_DELAY' : 2#add delay because otheriwse 429 unknown status
    })

    process.crawl(CredihealthSpider)
    process.start()

if __name__ == '__main__':
    run_spider()





