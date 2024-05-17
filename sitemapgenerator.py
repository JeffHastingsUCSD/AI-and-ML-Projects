import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urljoin, urlparse
import xml.dom.minidom as minidom

class SitemapGenerator:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.visited = set()
        self.to_visit = [self.base_url]
        self.sitemap_urls = []

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def fetch_links(self, url):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href")
                if href == "" or href is None:
                    continue
                href = urljoin(url, href)
                href = urlparse(href)._replace(fragment='').geturl()
                if self.is_valid_url(href) and href.startswith(self.base_url):
                    links.append(href)
            return links
        except Exception as e:
            print(f"Error fetching links from {url}: {e}")
            return []

    def crawl(self):
        while self.to_visit:
            url = self.to_visit.pop(0)
            if url not in self.visited:
                print(f"Visiting: {url}")
                self.visited.add(url)
                self.sitemap_urls.append(url)
                links = self.fetch_links(url)
                for link in links:
                    if link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)
                print(f"Found {len(links)} links on {url}")

    def create_sitemap(self, file_path='sitemap.xml'):
        urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

        for url in self.sitemap_urls:
            url_elem = ET.SubElement(urlset, "url")

            loc = ET.SubElement(url_elem, "loc")
            loc.text = url

            lastmod = ET.SubElement(url_elem, "lastmod")
            lastmod.text = datetime.now().strftime("%Y-%m-%d")

            changefreq = ET.SubElement(url_elem, "changefreq")
            changefreq.text = "weekly"

            priority = ET.SubElement(url_elem, "priority")
            priority.text = "0.5"

        tree = ET.ElementTree(urlset)
        self.pretty_print_xml(urlset)
        tree.write(file_path, xml_declaration=True, encoding='utf-8', method="xml")
        print(f"Sitemap successfully created at {file_path}")

    def pretty_print_xml(self, elem):
        """Pretty prints XML to the console."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml_as_string = reparsed.toprettyxml(indent="  ")
        print(pretty_xml_as_string)

# Example usage
base_url = 'https://www.facebook.com'
generator = SitemapGenerator(base_url)
generator.crawl()
generator.create_sitemap('sitemap.xml')
