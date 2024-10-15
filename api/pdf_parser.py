# Imports
#================================================================================================
#================================================================================================

from datetime import datetime
import time

import requests
from pathlib import Path
import pymupdf as mupdf
import pymupdf4llm as mupdf4llm
import urllib
import numpy as np
# import boto3
# import os
# import io
# import urllib3
import logging

# Add the parent directory to sys.path
# import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from markdownify import markdownify as md
from urllib.parse import urlparse
from bs4 import BeautifulSoup
# from tqdm import tqdm
from typing import List, Tuple
from dotenv import load_dotenv
from api.dbutils.scrapping_utils import get_user_agent
from api.dbutils.data_validation import contentModel, attachementModel

# Set up logging
#================================================================================================
#================================================================================================


DIR = Path(__file__).parent.absolute()

load_dotenv()

# # Ensure the log directory exists
# log_dir = DIR / "output" / "log"
# log_dir.mkdir(parents=True, exist_ok=True)

# # Set up logging
# log_file = log_dir / "attachment_parser.log"
# log_file.touch(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


# PDFExtractor class
#================================================================================================
#================================================================================================

class PDFExtractor:
    """Class to extract text and images from a PDF
    """
    def __init__(self):
        self.output_path = DIR / "output"
        self.createDate = None
        self.modDate = None
        self.pdf_dims = None
        self.filetype = None
        self.filename = None
        self.file_size = None
        self.num_pages = None
        self.url = None
        self.content_model = None
        self.attachment_model = None
        self.time_taken = None


    def __repr__(self) -> str:
        return (
            f"PDFExtractor(\n"
            f"  time_taken: {round(self.time_taken,2)},\n"
            f"  url: {self.url},\n"
            f"  creation_date: {self.createDate},\n"
            f"  modified_date: {self.modDate},\n"
            f"  filename: {self.filename},\n"
            f"  filetype: {urllib.parse.unquote(self.filetype)},\n"
            f"  file_size: {round(self.file_size,2)} MB,\n"
            f"  pdf_dims: {self.pdf_dims},\n"
            f"  num_pages: {self.num_pages},\n"
            f"  content_model: {self.content_model},\n"
            f"  attachment_model: {self.attachment_model}\n"
            f")"
        )
    
    def get_metadata(self,doc:mupdf.Document):
        # Get Metadata
        self.createDate = datetime.strptime(doc.metadata['creationDate'][2:16], '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%SZ') if ('creationDate' in doc.metadata and doc.metadata['creationDate']) else None
        self.modDate = datetime.strptime(doc.metadata['modDate'][2:16], '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%SZ') if ('modDate' in doc.metadata and doc.metadata['modDate']) else None
        self.pdf_dims = f"{round(doc[0].rect.width)}x{round(doc[0].rect.height)}"
        self.num_pages = len(doc)
        self.filetype = doc.metadata['format'].split()[0].lower()
        self.filename = (doc.metadata['title'] if doc.metadata['title'] else datetime.now().strftime('%Y%m%d%H%M%S')+ f".{self.filetype}")
        

    def get_pdf(self, link: str) -> mupdf.Document:
        """Download PDF from a given URL

        Args:
            link (str): URL of the PDF

        Returns:
            Path: Path to the downloaded PDF
        """
        self.url = link
        
        parsed_url = urlparse(link)
        
        header = {
            "User-Agent": get_user_agent(),
            "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": f"{parsed_url.scheme}://{parsed_url.netloc}",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Try to get response from the URL
        try:
            logging.info(f"Downloading PDF: {link}")
            response = requests.get(link, headers=header ,verify=False, timeout=500 ,stream=True)
        except Exception as e:
            logging.error(f"Error downloading PDF: {e} \n Writing to problematic_pdf_link.txt")
            ## Write problematic links to a file
            # with open(DIR/"output"/"problematic_links.text","a+") as f:
            #     f.seek(0)
            #     existing_links = f.read().splitlines()
            #     if link not in existing_links:
            #         f.write(link + "\n")
                    
            return None
        else:
            ## Check if the response is successful
            if response.status_code != 200:
                logging.error(f"Error downloading PDF, Response code: {response.status_code}")
                return None
        
        # Stream the PDF content to PyMuPDF
        try:
            doc = mupdf.Document(stream=response.content,filetype="pdf")
            # Get Metadata
            self.get_metadata(doc)
            self.file_size = len(response.content)
            
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            return None
        else:
            logging.info(f"Successfully downloaded PDF: {self.filename}")
            return doc
    
    
    def create_clipping_rectangle(self,page: mupdf.Page) -> mupdf.Rect:
        """Create a clipping rectangle for a given page, made using vectorization for speed.

        Args:
            page (mupdf.Page): Page object from PyMuPDF

        Returns:
            mupdf.Rect: Clipping rectangle
            
        Examples:
            >>> create_clipping_rectangle(page)
            mupdf.Rect(0, 0, 595.32, 841.92)
        """
        # Convert the page to an image object
        pix = page.get_pixmap()
        
        #Change the image to numpy array immediately rather than converting to PIL image
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # Convert to grayscale
        #Values are weighted to account for human perception of color
        # Retunrs a 2D array of the image in grayscale
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Get image dimensions
        height, width = img_gray.shape
        
        # Define thresholds
        color_threshold = 250 # Higher values are closer to white
        content_threshold = 0.10 # More than 20% non-white pixels to be considered content
        margin = int(height * 0.08) # 8% of page height as margin
        
        # Vectorized row content check
        # Check if the mean of the row is less than the color threshold
        # If the mean is greater than the content threshold, the row is considered content
        # This is like a for loop but much faster,it checks all elements in the row at once
        row_content = np.mean(img_gray < color_threshold, axis=1) > content_threshold
        # print(row_content)
        
        # Vectorized fully colored check
        # Check if the mean of the row is equal to 1.0
        # If the mean is 1.0, the row is fully colored
        # This is like a for loop but much faster,it checks all elements in the row at once
        fully_colored = np.mean(img_gray < color_threshold, axis=1) == 1.0
        
        # Check for fully colored section and adjust start_y if found
        # margin refers to position of the top margin
        # checks if the position of the top margin is fully colored by checking if any row in the fully_colored array is True
        if np.any(fully_colored[margin]):
            start_y = int(height * 0.20) # Increase margin to 10% if fully colored section is detected
        else:
            start_y = margin
        
        # Find the first content row after the top margin
        start_y = np.argmax(row_content[start_y:]) + start_y
        
        # Find the last content row before the bottom margin
        end_y = height - margin - np.argmax(row_content[start_y:height-margin][::-1]) - 5
        
        # Add safety margin
        safety_margin = 37
        start_y = max(0, start_y - safety_margin)
        end_y = min(height, end_y + safety_margin)
        
        # Create and return the clipping rectangle
        return mupdf.Rect(0, start_y, width, end_y)
    
    
    def extract_text_with_clipping(self,doc:mupdf.Document) -> Tuple[List[str], List[str], List[str]]:
        """Extract text from a PDF with clipping, returning lists of lines for each page.

        Args:
            pdf_path (Path): Path to the PDF

        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple containing lists of plain text, HTML, and markdown

        Example usage:
        >>> plain, html, markdown = extract_text_with_clipping(pdf_path)
        >>> print(plain[0])  # First line of the first page
        "This is the first line of the first page of the PDF"
        >>> print(html[0])  # First line of HTML from the first page
        "<p>This is the first line of the first page of the PDF</p>"
        >>> print(markdown[0])  # First line of markdown from the first page
        "# This is the first line of the first page of the PDF"
        """
        plain, html, markdown = [], [], []
        
        if doc is None:
            return plain, html, markdown

        for page in doc:
            clip_rect = self.create_clipping_rectangle(page)
            
            for i,tab in enumerate(page.find_tables(strategy='lines_strict')):  # iterate over all tables
                page.add_redact_annot(tab.bbox)
            
            page.apply_redactions()  # erase all table text
            
            # Extract plain text and split into lines
            page_text = page.get_text(clip=clip_rect).splitlines()
            plain.append([line.strip() for line in page_text if line.strip()])

            # Extract HTML and split into lines, filter out image tags
            html_text = page.get_text('html', clip=clip_rect).splitlines()
            soup = BeautifulSoup("\n".join(html_text), 'html.parser')
            # Get rid of images in the HTML
            for img in soup.find_all('img'):
                img.decompose()
            html.append([line.strip() for line in soup.prettify().splitlines() if line.strip()])

        # Extract markdown
        markdown_text = mupdf4llm.to_markdown(doc, show_progress=False,write_images=False,table_strategy="lines_strict",margins=(70,70))
        markdown = [page.splitlines() for page in markdown_text.split('\f') if page.strip()]
        
        #Flatten the lists
        plain = [line for page in plain for line in page]
        html = [line for page in html for line in page]
        markdown = [line for page in markdown for line in page]

        return plain, html, markdown
    
    
    def extract_images_with_clipping(self,doc:mupdf.Document,upload:bool=False) -> List[Tuple[str,str]]:
        """Extract images from a PDF with clipping, returning a list of image URLs and names.

        Args:
            doc (mupdf.Document): PDF document object
            upload (bool, optional): Upload images to S3. Defaults to False.

        Returns:
            List[Tuple[str,str]]: List of tuples containing image URLs and names
        
        Example usage:
        >>> images = extract_images_with_clipping(doc)
        >>> print(images[0])
        ("https://s3.amazonaws.com/bucket/image.jpg","image.jpg")
        """
        image_urls = []
        image_names = []
        #Image dims check
        dimlimit = 150  # 100  # each image side must be greater than this
        relsize = 0.10  # 0.05  # image : image size ratio must be larger than this (10%)
        
        # Helper functions
        # @staticmethod
        # def upload_to_bucket(file_name:str, file_data:object,upload:bool=upload):
        #     s3 = boto3.client(
        #         service_name="s3",
        #         endpoint_url=os.getenv("ENDPOINT_URL"),
        #         aws_access_key_id=os.getenv("AWS_KEY"),
        #         aws_secret_access_key=os.getenv("AWS_SECRET"),
        #         region_name="auto",
        #     )
            
        #     # Upload/Update single file
        #     if upload:
        #         s3.upload_fileobj(file_data, "siaran-temp", file_name)
            
        #     return f"{os.getenv("S3_URL")}{file_name}"
        
        @staticmethod
        def recoverpix(doc:mupdf.Document, item:List)->dict:
            """Recover image data from a PDF document.Will also handle images with transparent backgrounds.

            Args:
                doc (mupdf.Document): PDF document object
                item (List): List of image data

            Returns:
                image_data (Dict): Image data
            """
            
            xref = item[0]  # xref of PDF image
            smask = item[1]  # xref of its /SMask

            # special case: /SMask or /Mask exists
            if smask > 0:
                pix0 = mupdf.Pixmap(doc.extract_image(xref)["image"])
                if pix0.alpha:  # catch irregular situation
                    pix0 = mupdf.Pixmap(pix0, 0)  # remove alpha channel
                mask = mupdf.Pixmap(doc.extract_image(smask)["image"])

                try:
                    pix = mupdf.Pixmap(pix0, mask)
                except:  # fallback to original base image in case of problems
                    pix = mupdf.Pixmap(doc.extract_image(xref)["image"])

                if pix0.n > 3:
                    ext = "pam"
                else:
                    ext = "png"

                return {  # create dictionary expected by caller
                    "ext": ext,
                    "colorspace": pix.colorspace.n,
                    "image": pix.tobytes(ext),
                }

            # special case: /ColorSpace definition exists
            # to be sure, we convert these cases to RGB PNG images
            if "/ColorSpace" in doc.xref_object(xref, compressed=True):
                pix = mupdf.Pixmap(doc, xref)
                pix = mupdf.Pixmap(mupdf.csRGB, pix)
                return {  # create dictionary expected by caller
                    "ext": "png",
                    "colorspace": 3,
                    "image": pix.tobytes("png"),
                }
                
            return doc.extract_image(xref)
                
        for i, page in enumerate(doc):
            img_list = doc.get_page_images(i)
            # xrefs = [x[0] for x in img_list]
            
            for img in img_list:
                
                width = img[2]
                height = img[3]
                if min(width, height) <= dimlimit:
                    continue
                
                image = recoverpix(doc, img)
                imgdata = image["image"]
                
                if len(imgdata) / (width * height * image['colorspace']) <= relsize:
                        continue
                
                img_name = f"{self.filename.split('.')[0]}_img_{i}.png"
                image_names.append(img_name)
                
                # Upload image data directly to S3
                # with io.BytesIO(imgdata) as img_buffer:
                #     image_urls.append(upload_to_bucket(img_name, img_buffer))
                    
                with open(self.input_path / img_name, "wb") as f:
                    f.write(imgdata)
            
        return zip(image_urls,image_names)


    def create_content_model(self, plain: List[str], html: List[str], markdown: List[str]) -> contentModel:
        """Create a ContentModel object to be used with SiaranModel

        Args:
            plain (List[str]): plain text extracted from the PDF
            html (List[str]): html text extracted from the PDF
            markdown (List[str]): markdown text extracted from the PDF

        Returns:
            ContentModel: ContentModel object
        """
        if len(plain) > 1:
            return contentModel(plain=plain, html=html, markdown=markdown)
        else:
            return contentModel(plain=[], html=[], markdown=[])


    def create_attachment_model(self,doc:mupdf.Document,upload_images:bool=False) -> List[attachementModel]:
        """
        Create an AttachmentModel object to be used with SiaranModel
        - If upload_images is True, images will be uploaded to S3 and URLs will be returned
        - If Doc is None, an empty list will be returned


        Returns:
            List[AttachmentModel]: List of AttachmentModel objects if doc is not None
        """
        
        attachments = []
        
        if doc is None:
            return attachments
        
        # Only extract images if the PDF has images
        try:
            # Extract images and create image attachments
            images = self.extract_images_with_clipping(doc,upload=upload_images)
            image_attachments = [
                attachementModel(
                url=url,
                alt=None,
                file_name=file_name,
                file_type='image'
                ) for url,file_name in images
            ]
            
        except NameError as e:
            print(f"No images found in PDF")
        except Exception as e:
            print(f"Error extracting images: {e}")
        else:
            attachments.extend(image_attachments)

        # Add the PDF as an attachment
        pdf_attachment = attachementModel(
            url=self.url,
            alt=None,
            file_name=self.filename,
            file_type=self.filetype
        )
        attachments.append(pdf_attachment)

        return attachments
    
    
    def write_to_files(self):
        """Write extracted text and images to files

        Args:
            plain (List[str]): List of plain text
            html (List[str]): List of HTML text
            markdown (List[str]): List of markdown text
        """
        # Write text to files
        
        file_contents = {
            f"{self.filename.split('.')[0]}_plain.txt": "\n".join(self.content_model.plain),
            f"{self.filename.split('.')[0]}_html.html": "\n".join(self.content_model.html),
            f"{self.filename.split('.')[0]}_markdown.md": "\n".join(self.content_model.markdown)
        }

        for file_name, content in file_contents.items():
            with open(DIR/"output"/"files"/ file_name, "w") as f:
                f.write(content)


    def process_pdf(self, link:str, get_images:bool= False) -> List:
        """
        Process a PDF from a given URL
        - If link is not valid or does not return a PDF, ContentModel and AttachmentModel will be empty

        Args:
            link (str): URL of the PDF
            get_images (bool, optional): Extract images from the PDF,will also upload images to S3. Defaults to False.

        Returns:
            List[ContentModel,AttachmentModel]: List of ContentModel and AttachmentModel objects
        """
        doc = self.get_pdf(link)
        
        start_time = time.time()
        plain, html, markdown = self.extract_text_with_clipping(doc)
        # self.write_to_files(plain, html, markdown)
        self.content_model = self.create_content_model(plain, html, markdown)
        
        if get_images:
            self.attachment_model = self.create_attachment_model(doc, upload_images=True)
        
        end_time = time.time()
        self.time_taken = end_time - start_time
        
        logging.info(f"Successfully extracted contents of PDF: {self.filename}")
        logging.info(f"Time taken: {self.time_taken}")

        return self.content_model,self.attachment_model


# # Test the PDFExtractor class
# #================================================================================================
# #================================================================================================

# def main(links: List[str]):
#     link = "https://ekonomi.gov.my/sites/default/files/2023-05/Speech_Affin-Conference_30May2023.pdf"
#     extractor = PDFExtractor()
#     content_model,attachment_model = extractor.process_pdf(link,get_images=False)
#     extractor.write_to_files()
#     print("Time Taken: ",extractor.time_taken)
#     print(extractor.__repr__())

# if __name__ == "__main__":
#     from pymongo import MongoClient
    
#     links = []

#     client = MongoClient(os.getenv("DATABASE_URI"))
#     db = client[os.getenv("DB_NAME")]
#     collection = db[os.getenv("COLLECTION_NAME")]

#     pipeline = [
#         {"$match": {"attachments.file_type": "pdf"}},
#         {"$sample": {"size": 5}}
#     ]
#     random_entries = list(collection.aggregate(pipeline))
#     for entry in random_entries:
#         for attachment in entry.get('attachments', []):
#             if attachment.get('file_type') == 'pdf':
#                 links.append(attachment.get('url'))
    
#     main(links)

