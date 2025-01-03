# Imports
#================================================================================================
#================================================================================================

from datetime import datetime,timedelta,timezone
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
import re

from PIL import Image, ImageFilter
from urllib.parse import urlparse
from typing import List, Tuple,Union
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
        
        def date_parser(date:str)->str:
            formats = ['%Y%m%d%H%M%S', '%d/%m/%y %H:%M']
            for fmt in formats:
                try:
                    return datetime.strptime(date, fmt).strftime('%Y-%m-%dT%H:%M:%SZ')
                except ValueError:
                    continue
            return None
            
        # Get Metadata
        self.createDate = date_parser(doc.metadata['creationDate'][2:16]) if ('creationDate' in doc.metadata and doc.metadata['creationDate']) else None
        self.modDate = date_parser(doc.metadata['modDate'][2:16]) if ('modDate' in doc.metadata and doc.metadata['modDate']) else None
        self.pdf_dims = f"{round(doc[0].rect.width)}x{round(doc[0].rect.height)}"
        self.num_pages = len(doc)
        self.filetype = doc.metadata['format'].split()[0].lower()
        self.filename = (doc.metadata['title'] if doc.metadata['title'] else datetime.now().strftime('%Y%m%d%H%M%S')+ f".{self.filetype}")


    # def _show_image(self,page, title="",grayscale=False):
    #     """Display a pixmap.

    #     Just to display Pixmap image of "item" - ignore the man behind the curtain.

    #     Args:
    #         item: any PyMuPDF object having a "get_pixmap" method.
    #         title: a string to be used as image title

    #     Generates an RGB Pixmap from item using a constant DPI and using matplotlib
    #     to show it inline of the notebook.
    #     """
    #     DPI = 150  # use this resolution
    #     import numpy as np

    #     # %matplotlib inline
    #     pix = page.get_pixmap(dpi=DPI)
    #     img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
    #     plt.figure(dpi=DPI)  # set the figure's DPI
    #     plt.title(title)  # set title of image
    #     if grayscale:
    #         img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    #         plt.imshow(img,cmap='gray', extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))
    #     else:
    #         plt.imshow(img, extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))
        
    #     plt.show()
        

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
            mupdf.TOOLS.set_small_glyph_heights(True)
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
    
    def _create_clipping_rectangle(
            self, 
            page: mupdf.Page,
            content_threshold:float = 0.08,
            color_threshold:float = 0.15,
            std_threshold:float = 0.1,
            margin_ratio:float = 0.08,
            color_row_threshold:float = 0.15,
            window_detection_size: Union[int, Tuple[int, int]] = (20, 20)
    ) -> mupdf.Rect:
        """Create a clipping rectangle for a given page using enhanced image processing.
        
        The function processes the image to enhance text-like features and detect colored
        regions that might indicate headers and footers, even if not fully colored.

        Args:
            page (mupdf.Page): Page object
            content_threshold (float, optional): Threshold for content detection. Defaults to 0.08.
            color_threshold (float, optional): Threshold for color detection. Defaults to 0.15.
            std_threshold (float, optional): Threshold for standard deviation detection. Defaults to 0.1.
            margin_ratio (float, optional): Ratio of the margin to the height of the page. Defaults to 0.08.
            color_row_threshold (float, optional): Threshold for colored row detection. Defaults to 0.15.
            window_detection_size (Union[int, Tuple[int, int]], optional): Size of the window for detection. Defaults to (20, 20).
        
        Returns:
            mupdf.Rect: Clipping rectangle
        """
        ## Delete any images 
        for img in page.get_images():
            try:
                page.delete_image(img[0])
            except ValueError:
                pass
        
        ## Convert the page to an image array
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        height, width = img_array.shape[0], img_array.shape[1]
        
        ## Set the parameters
        content_threshold = content_threshold
        color_threshold = color_threshold
        margin = int(height * margin_ratio)
        color_row_threshold = color_row_threshold
        
        ## Set the window detection size
        if isinstance(window_detection_size, int):
            window_detection_size = (window_detection_size, window_detection_size)
        
        def _detect_colored_regions(img_array: np.ndarray) -> np.ndarray:
            """Detect colored regions within an image by analyzing RGB color patterns.
            This function identifies colored regions in an image by examining two key metrics:
            1. Color saturation - calculated as (max_RGB - min_RGB)/max_RGB
            2. RGB channel standard deviation - measures color variation across channels
            The detection is based on thresholding both metrics to identify pixels that likely
            contain meaningful color information rather than grayscale content.
                img_array (np.ndarray): 3D numpy array of shape (height, width, 3) containing 
                    the RGB image data with values in range [0, 255]
                np.ndarray: 1D array of shape (height,) containing the fraction of colored pixels
                    in each row. Values range from 0.0 (no colored pixels) to 1.0 (all pixels colored).
            Note:
                The function uses two global thresholds:
                - color_threshold: Minimum saturation value to consider a pixel as colored
                - std_threshold: Minimum RGB standard deviation to consider a pixel as colored
                A pixel is considered colored if it exceeds either threshold
                
            Args:
                img_array (np.ndarray): Image array

            Returns:
                np.ndarray: Array of colored rows
            """
            # Normalize RGB values to [0,1] range
            img_normalized = img_array.astype(float) / 255.0
            
            # Get max and min RGB values for each pixel
            max_rgb = np.max(img_normalized, axis=2)
            min_rgb = np.min(img_normalized, axis=2)
            
            # Small value to avoid division by zero
            eps = 1e-8
            
            # Calculate saturation as (max-min)/max for each pixel
            # Use np.divide with out parameter to handle division by zero
            saturation = np.divide(
                max_rgb - min_rgb, 
                max_rgb + eps, 
                out=np.zeros_like(max_rgb), 
                where=max_rgb != 0
            )
            
            # Calculate standard deviation across RGB channels
            rgb_std = np.std(img_normalized, axis=2)
            
            # Detect colored pixels where either:
            # - Saturation exceeds color_threshold
            # - RGB standard deviation exceeds std_threshold
            is_colored = (saturation > color_threshold) | (rgb_std > std_threshold)
            
            # Calculate fraction of colored pixels in each row
            colored_rows = np.mean(is_colored, axis=1)
            
            return colored_rows
        
        def _enhance_text_regions(img_array: np.ndarray) -> np.ndarray:
            """Enhance text regions in the image array using dilation and smoothing techniques.
            This function performs several image processing steps:
            1. Binarizes the image using a threshold of 248
            2. Applies horizontal dilation with a kernel size of (1,5)
            3. Applies vertical dilation with a kernel size of (3,1)
            4. Smooths the image using Gaussian blur with radius 1
            5. Normalizes the final image to [0,1] range
                img_array (np.ndarray): Input image array, typically grayscale
                np.ndarray: Normalized image array with enhanced text regions, values in range [0,1]
            Note:
                This function is particularly useful for improving text detection in documents
                by making text regions more prominent and reducing noise.

            Args:
                img_array (np.ndarray): img_array array

            Returns:
                normalized (np.ndarray): Normalized img_array array
            """
            
            # Convert the image to binary by thresholding
            binary = img_array < 248
            
            # Apply horizontal dilation
            horizontal_kernel = np.ones((1, 5))
            dilated_horizontal = np.apply_along_axis(lambda m: np.convolve(m, horizontal_kernel.flatten(), mode='same'), axis=1, arr=binary.astype(np.uint8))
            
            # Apply vertical dilation
            vertical_kernel = np.ones((3, 1))
            dilated = np.apply_along_axis(lambda m: np.convolve(m, vertical_kernel.flatten(), mode='same'), axis=0, arr=dilated_horizontal)
            
            # Smooth the image using Gaussian blur
            smoothed = np.array(Image.fromarray(dilated).convert("L").filter(ImageFilter.GaussianBlur(radius=1)))
            
            # Normalize the image to the range [0, 1]
            normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
            return normalized
        
        # 1. Convert the image to grayscale
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        
        # 2. Enhance text regions and detect colored regions
        enhanced_img = _enhance_text_regions(img_gray)
        colored_rows = _detect_colored_regions(img_array)
        
        # 3. Detect content and colored regions
        ## Detect content regions
        row_content = np.mean(enhanced_img > 0.5, axis=1) >= content_threshold
        ## Detect colored regions
        colored_regions = colored_rows >= color_row_threshold
        ## Expand and smooth the colored regions
        colored_regions = np.apply_along_axis(lambda m: np.convolve(m, np.ones(5), mode='same'), axis=0, arr=colored_regions.astype(np.uint8))
        colored_regions = np.apply_along_axis(lambda m: np.convolve(m, np.ones(3), mode='same'), axis=0, arr=colored_regions)
        
        
        # 5. Detect the start and end of the content region
        start_y = margin // 2 
        while start_y < height - margin:
            window_size = window_detection_size[0]
            window_end = min(start_y + window_size, height)
            # Check if the window contains colored regions or only content
            if np.all(colored_regions[start_y:window_end]) or np.all(row_content[start_y:window_end]):
                start_y += int(height * 0.08)
            else:
                break
        
        ## Set end_y to the bottom of the page by a margin
        end_y = height - margin // 2
        while end_y > start_y:
            window_size = window_detection_size[-1]
            window_start = max(end_y - window_size, 0)
            # Check if the window contains colored regions or only content
            if np.any(colored_regions[window_start:end_y]) or np.all(row_content[window_start:end_y]):
                end_y -= int(height * 0.15)
            else:
                break
        
        # 6. Find the start and end of the content region
        content_start = np.argmax(row_content[start_y:end_y]) + start_y if np.any(row_content[start_y:end_y]) else start_y
        content_slice = row_content[content_start:end_y]
        content_end = end_y - np.argmax(content_slice[::-1]) if np.any(content_slice) else end_y
        
        # 7. Create the clipping rectangle
        safety_margin = int(height * 0.025)
        final_start = max(0, content_start - safety_margin)
        final_end = min(height, content_end + safety_margin)
        
        return mupdf.Rect(0, final_start, width, final_end)


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
        >>> print(markdown[0])  # First line of markdown from the first page
        "# This is the first line of the first page of the PDF"
        """
        def span_prep(block_dict):
            def block_prep(block_list_dict):
                # print(block_list_dict)
                span_y1 = None
                span_curr = None
                all_span = []
                for enumspan, span in enumerate(block_list_dict["lines"]):
                    # print(span)
                    if span["spans"]:
                        if enumspan == 0:
                            span_y1 = span["spans"][0]["bbox"][3]
                            span_curr = [span]
                        else:
                            if span["spans"][0]["bbox"][1] > span_y1:
                                # print("nl")
                                all_span.append(span_curr)
                                span_curr = [span]
                            else:
                                span_curr.append(span)
                            span_y1 = span["spans"][0]["bbox"][3]
                all_span.append(span_curr)
                return all_span
            all_span = []
            for bbi in block_dict["blocks"]:
                all_span += block_prep(bbi)
            line_fixed = []
            for span_line in all_span:
                if span_line:
                    all_bbox = [x["bbox"] for i in span_line for x in i["spans"]]
                    max_bbox = (
                        min(all_bbox)[0],
                        min(all_bbox)[1],
                        max(all_bbox)[2],
                        max(all_bbox)[3],
                    )
                    new_line_dict = {"bbox": max_bbox, "lines": span_line}
                    line_fixed.append(new_line_dict)
            return line_fixed

        def markdown_formatting(text, flag, type):
            def pattern_fix(m):
                return "\n" + m.group(0)

            bullet_patterns = r"(^)[●|•|○|·|◦|‣|∙|§||]|\\uf0b7|Ø(\s|$)" # Matches common bullet point characters

            numbered_list_pattern = [
                r'(^)(\s?)\d{1,3}\.(\s|$)',   # Matches numbered lists
                r'(^)(\s?)\d{1,3}\)(\s|$)',   # Matches numbered lists
                r'(^)(\s?)[XVI|xvi]{1,4}\.(\s|$)',  # Matches alphabetical lists
                r'(^)(\s?)[XVI|xvi]{1,4}\)(\s|$)',  # Matches alphabetical lists
            ]

            cleaned_text = cleaned_text_plain = text
            cleaned_text = re.sub(r"(\_|\—){3,}", "", cleaned_text)
            cleaned_text = cleaned_text.replace("*",r"\*")
            cleaned_text = cleaned_text.replace("_",r"\_")

            if re.search(bullet_patterns, cleaned_text):
                cleaned_text = re.sub(f"{bullet_patterns}", "\n- ", cleaned_text)
                cleaned_text_plain = re.sub(f"{bullet_patterns}", "\n- ", cleaned_text_plain)

            if any(re.search(pattern, cleaned_text) for pattern in numbered_list_pattern):
                for pattern_ in numbered_list_pattern:
                    if re.search(pattern_, cleaned_text):
                        cleaned_text = re.sub(f"{pattern_}", pattern_fix,cleaned_text)
                        cleaned_text_plain = re.sub(f"{pattern_}", pattern_fix,cleaned_text_plain)
                        continue

            if flag & 2**4:
                cleaned_text = f"**{cleaned_text}**"
                cleaned_text = re.sub(r'^\*\* ', " **", cleaned_text)
                cleaned_text = re.sub(r'[\s]{1,}\*\*(\s?)*$', "** ", cleaned_text)

            if flag & 2**1:
                cleaned_text = f"_{cleaned_text}_"
                cleaned_text = re.sub(r'^\_ ', " _", cleaned_text)
                cleaned_text = re.sub(r' \_$', "_ ", cleaned_text)
                cleaned_text = re.sub(r'[\s]{1,}\_(\s?)*$', "_ ", cleaned_text)

            if type == "plain":
                return cleaned_text_plain
            elif type == "markdown":
                return cleaned_text
            
        def proc_text(line_fixed):
            line_y = None
            line_width = None
            spacing_width = None
            prev_spacing_width = 0
            line_spacing = 0.9
            block_text = []
            block_text_md = []
            line_text_plain = []
            line_text_md = []
            for text_line in line_fixed:

                if (
                    len(text_line["lines"]) == 1
                    and not text_line["lines"][0]["spans"][0]["text"].strip()
                ):
                    continue
                plain_str = " ".join(
                    [
                        markdown_formatting(x["text"], x["flags"], "plain")
                        for i in text_line["lines"]
                        for x in i["spans"]
                    ]
                )
                markdown = " ".join(
                    [
                        markdown_formatting(x["text"], x["flags"], "markdown")
                        for i in text_line["lines"]
                        for x in i["spans"]
                    ]
                )

                if not plain_str.strip():
                    continue

                prev_spacing_width = spacing_width
                line_width = round(text_line["bbox"][3] - text_line["bbox"][1], 0)

                if line_y:
                    if round(text_line["bbox"][1] - line_y[1], 0) > 0:
                        spacing_width = round(text_line["bbox"][1] - line_y[1], 0)
                    elif round(text_line["bbox"][1] - line_y[1], 0) <= 0:
                        # print(text_line["bbox"][3] - text_line["bbox"][1])
                        spacing_width = round(
                            (text_line["bbox"][3] - text_line["bbox"][1]) * 0.3, 0
                        )
                else:
                    spacing_width = round(line_width * line_spacing, 0)

                line_y = (text_line["bbox"][1], text_line["bbox"][3])

                if prev_spacing_width:
                    if ((spacing_width- prev_spacing_width)<= 1 )and spacing_width <= round(
                        line_width * line_spacing, 0
                    ):
                        # print("same line")
                        line_text_plain.append(plain_str)
                        line_text_md.append(markdown)
                    elif line_text_plain:
                        block_text.append(line_text_plain)
                        block_text_md.append(line_text_md)
                        line_text_plain = [plain_str]
                        line_text_md = [markdown]
                else:
                    line_text_plain.append(plain_str)
                    line_text_md.append(markdown)

            block_text.append(line_text_plain)
            block_text_md.append(line_text_md)

            return block_text, block_text_md

        def clean_regex(text):
            def clean_num_bold(t):
                return t.group(0)[3:] + "**"
            def clean_num_italics(t):
                return t.group(0)[2:] + "_"
            regex_checklist = [
                (r"\u202c\u202d", " "),
                (r"\u202c|\u202d", ""),
                (" \r  ", " "),
                (r"\u2010|\u00ad",""),
                (r"[\*]{2}\n\d{1,3}[\.|\)](\s|[\*]{2})", clean_num_bold),
                (r"\_\n\d{1,3}[\.|\)](\s|\_)", clean_num_italics),
                (r"[\*]{2}\n\d{1,3}[\.|\)]\s", clean_num_bold),
                (r"\_\n\d{1,3}[\.|\)\s]", clean_num_italics),
                (r"[\*]{4}", ""),
                (r"[\*]{2}\.", "**."),
                (r"[\*]{2}\s*[\*]{2}", " "),
                (r"[\_]{2}", ""),
                (r"[\_]{1}\.", "_."),
                (r"[\_]{1}\s*[\_]{1}", " "),
                (r"[\n]{2,}",r"\n\n"),
                (r"[ ]{2,}"," "),
            ]

            for i in regex_checklist:
                text = re.sub(i[0],i[1], text)
            
            return text.strip()

        if doc is None:
            return [], []

        text_list = []
        text_list_md = []
        plain = []
        markdown = []

        for page in doc:
            clip_rect = self._create_clipping_rectangle(page)
            page.add_rect_annot(clip_rect)
            
            for i,tab in enumerate(page.find_tables(strategy='lines_strict')):  # iterate over all tables
                if clip_rect.contains(tab.bbox):
                    page.add_redact_annot(tab.bbox)
                    
            #angle check
            words_dir = page.get_textpage(flags=8+32+1+2).extractDICT()['blocks']
            for block in words_dir:
                for line in block['lines']:
                    angle = round(np.degrees(np.arctan2(line['dir'][1], line['dir'][0])))
                    if angle not in range(-5,6):
                        page.add_redact_annot(mupdf.Rect(*line['bbox']))
            
            page.apply_redactions()  # erase all table text
            
            # # #? Debugging
            # self._show_image(page, title=f"Page",grayscale=False)
            
            # for page_num, page in enumerate(doc):
            dict_test = page.get_textpage(clip_rect).extractDICT(sort=True)
            # print(dict_test)

            line_fixed = span_prep(dict_test)
            if line_fixed:
                plain_block,md_block = proc_text(line_fixed)
                text_list += plain_block
                text_list_md += md_block

                plain = [clean_regex(" ".join(line)) for line in text_list ]
                markdown = [clean_regex(" ".join(line)) for line in text_list_md ]
        return plain,markdown
    
    #! Deprecated
    # def extract_images_with_clipping(self,doc:mupdf.Document,upload:bool=False) -> List[Tuple[str,str]]:
    #     """Extract images from a PDF with clipping, returning a list of image URLs and names.

    #     Args:
    #         doc (mupdf.Document): PDF document object
    #         upload (bool, optional): Upload images to S3. Defaults to False.

    #     Returns:
    #         List[Tuple[str,str]]: List of tuples containing image URLs and names
        
    #     Example usage:
    #     >>> images = extract_images_with_clipping(doc)
    #     >>> print(images[0])
    #     ("https://s3.amazonaws.com/bucket/image.jpg","image.jpg")
    #     """
    #     # Initialize lists
    #     image_urls = []
    #     image_names = []
        
    #     #Image dims check
    #     dimlimit = 150  # 100  # each image side must be greater than this
    #     relsize = 0.10  # 0.05  # image : image size ratio must be larger than this (10%)
        
    #     # Helper functions
    #     @staticmethod
    #     def upload_to_bucket(file_name:str, file_data:object,upload:bool=upload):
    #         s3 = boto3.client(
    #             service_name="s3",
    #             endpoint_url=os.getenv("ENDPOINT_URL"),
    #             aws_access_key_id=os.getenv("AWS_KEY"),
    #             aws_secret_access_key=os.getenv("AWS_SECRET"),
    #             region_name="auto",
    #         )
            
    #         # Upload/Update single file
    #         if upload:
    #             s3.upload_fileobj(file_data, "siaran-temp", file_name)
            
    #         return f"{os.getenv("S3_URL")}{file_name}"
        
    #     @staticmethod
    #     def recoverpix(doc:mupdf.Document, item:List)->dict:
    #         """Recover image data from a PDF document.Will also handle images with transparent backgrounds.

    #         Args:
    #             doc (mupdf.Document): PDF document object
    #             item (List): List of image data

    #         Returns:
    #             image_data (Dict): Image data
    #         """
            
    #         xref = item[0]  # xref of PDF image
    #         smask = item[1]  # xref of its /SMask

    #         # special case: /SMask or /Mask exists
    #         if smask > 0:
    #             pix0 = mupdf.Pixmap(doc.extract_image(xref)["image"])
    #             if pix0.alpha:  # catch irregular situation
    #                 pix0 = mupdf.Pixmap(pix0, 0)  # remove alpha channel
    #             mask = mupdf.Pixmap(doc.extract_image(smask)["image"])

    #             try:
    #                 pix = mupdf.Pixmap(pix0, mask)
    #             except:  # fallback to original base image in case of problems
    #                 pix = mupdf.Pixmap(doc.extract_image(xref)["image"])

    #             if pix0.n > 3:
    #                 ext = "pam"
    #             else:
    #                 ext = "png"

    #             return {  # create dictionary expected by caller
    #                 "ext": ext,
    #                 "colorspace": pix.colorspace.n,
    #                 "image": pix.tobytes(ext),
    #             }

    #         # special case: /ColorSpace definition exists
    #         # to be sure, we convert these cases to RGB PNG images
    #         if "/ColorSpace" in doc.xref_object(xref, compressed=True):
    #             pix = mupdf.Pixmap(doc, xref)
    #             pix = mupdf.Pixmap(mupdf.csRGB, pix)
    #             return {  # create dictionary expected by caller
    #                 "ext": "png",
    #                 "colorspace": 3,
    #                 "image": pix.tobytes("png"),
    #             }
                
    #         return doc.extract_image(xref)
                
    #     for i, page in enumerate(doc):
    #         img_list = doc.get_page_images(i)
    #         # xrefs = [x[0] for x in img_list]
            
    #         for img in img_list:
                
    #             width = img[2]
    #             height = img[3]
    #             if min(width, height) <= dimlimit:
    #                 continue
                
    #             image = recoverpix(doc, img)
    #             imgdata = image["image"]
                
    #             if len(imgdata) / (width * height * image['colorspace']) <= relsize:
    #                     continue
                
    #             img_name = f"{self.filename.split('.')[0]}_img_{i}.png"
    #             image_names.append(img_name)
                
    #             # Upload image data directly to S3
    #             with io.BytesIO(imgdata) as img_buffer:
    #                 image_urls.append(upload_to_bucket(img_name, img_buffer))
                    
    #             with open(self.input_path / img_name, "wb") as f:
    #                 f.write(imgdata)
            
    #     return zip(image_urls,image_names)


    def create_content_model(self, plain: List[str], markdown: List[str]) -> contentModel:
        """Create a ContentModel object to be used with SiaranModel

        Args:
            plain (List[str]): plain text extracted from the PDF
            html (List[str]): html text extracted from the PDF
            markdown (List[str]): markdown text extracted from the PDF

        Returns:
            ContentModel: ContentModel object
        """
        if len(plain) > 1:
            return contentModel(plain=plain, markdown=markdown)
        else:
            return contentModel(plain=[], markdown=[])


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
    
    
    def _write_to_files(self):
        """Write extracted text and images to files

        Args:
            plain (List[str]): List of plain text
            html (List[str]): List of HTML text
            markdown (List[str]): List of markdown text
        """
        # Write text to files
        
        file_contents = {
            f"{self.filename.split('.')[0]}_plain.txt": "\n".join(self.content_model.plain),
            # f"{self.filename.split('.')[0]}_html.html": "\n".join(self.content_model.html),
            f"{self.filename.split('.')[0]}_markdown.md": "\n".join(self.content_model.markdown)
        }

        for file_name, content in file_contents.items():
            with open(DIR + "output/" +file_name, "w") as f:
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
        plain, markdown = self.extract_text_with_clipping(doc)
        self.content_model = self.create_content_model(plain, markdown)
        
        if get_images:
            self.attachment_model = self.create_attachment_model(doc, upload_images=True)
        
        end_time = time.time()
        self.time_taken = end_time - start_time
        
        logging.info(f"Successfully extracted contents of PDF: {self.filename}")
        logging.info(f"Time taken: {self.time_taken}")

        return self.content_model,self.attachment_model


# Test the PDFExtractor class
#================================================================================================
#================================================================================================
# def main(links: List[str]):
#     link = "https://www.moh.gov.my/index.php/database_stores/attach_download/337/2114"
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