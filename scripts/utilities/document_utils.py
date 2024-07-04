from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

class DocumentProcessor:
    def is_epub_file(self, file_path):
        return file_path.lower().endswith('.epub')

    def extract_text_from_epub(self, file_path):
        book = epub.read_epub(file_path)
        text = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text.append(soup.get_text())
        return ' '.join(text)

    def split_text_into_sentences(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def create_documents(self, doc_path):
        if self.is_epub_file(doc_path):
            print(f"Processing EPUB file: {doc_path}")
            raw_text = self.extract_text_from_epub(doc_path)
            documents = self.split_text_into_sentences(raw_text)
        else:
            print(f"Processing text file: {doc_path}")
            with open(doc_path, "r") as file:
                raw_text = file.read()
            documents = self.split_text_into_sentences(raw_text)

        return documents