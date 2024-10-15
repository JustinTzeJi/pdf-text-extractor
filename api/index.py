# Imports
#================================================================================================
#================================================================================================
import urllib
from dbutils.data_validation import contentModel, attachementModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,HttpUrl
from pdf_parser import PDFExtractor
from typing import List,Optional


# Initialize the FastAPI app and models
#================================================================================================
#================================================================================================
app = FastAPI()

class PDFRequest(BaseModel):
    url: HttpUrl
    get_images: bool = False

class ContentSummary(BaseModel):
    plain_text_length: int
    html_length: int
    markdown_length: int
    num_attachments: int

class Metadata(BaseModel):
    url: str
    file_name:str
    file_type:str
    file_size:int
    num_pages:int
    creation_date:Optional[str]
    modified_date:Optional[str]
    processing_time:float
    content_summary:ContentSummary
    
class PDFResponse(BaseModel):
    metadata: Metadata
    content: contentModel
    attachments: Optional[List[attachementModel]]

class Response(BaseModel):
    success: bool
    data : PDFResponse

class Metadata_clone(BaseModel):
    url: str
    file_name:str
    file_type:str
    file_size:int
    num_pages:int
    creation_date:str
    modified_date:str


# API Endpoints
#================================================================================================
#================================================================================================
@app.get("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI"}

@app.post("/api/py/pdf_metadata")
async def pdf_metadata(request:PDFRequest)->dict:
    try:
        # Extract content from the PDF
        extractor = PDFExtractor()
        extractor.process_pdf(
            link = str(request.url), # change url from pydantic object to str since it expects a string
            get_images=request.get_images,
        )
        response = Metadata_clone(
                    url=str(request.url),
                    creation_date=extractor.createDate,
                    modified_date=extractor.modDate,
                    file_name=extractor.filename,
                    file_type=extractor.filetype,
                    file_size=extractor.file_size,
                    num_pages=extractor.num_pages,
                )
        return JSONResponse(
            status_code=200,
            content=response.model_dump()
        )

    except urllib.error.URLError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": 400,
                    "message": f"URL Error: {str(e)}"
                }
            }
        )
        
    except HTTPException as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": 400,
                    "message": f"Error processing PDF: {str(e.detail)}"
                }
            }
        )
        

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": f"Error processing PDF: {str(e)}"
                }
            }
        )
    

@app.post("/api/py/pdf_extract")
async def extract_pdf(request:PDFRequest)->dict:
    """ API endpoint to extract content from a PDF file

    Args:
        request (PDFRequest): Request object containing the URL of the PDF file

    Raises:
        HTTPException: If the PDF extraction fails

    Returns:
        dict: A dictionary containing the extracted content and attachment models
    """
    try:
        # Extract content from the PDF
        extractor = PDFExtractor()
        
        # Call the process_pdf method to extract content from the PDF
        content_model,attachment_model = extractor.process_pdf(
            link = str(request.url), # change url from pydantic object to str since it expects a string
            get_images=request.get_images,
        )

        # Raise an exception if the content model is None
        if not content_model or not content_model.plain:
            raise HTTPException(status_code=400, detail="Extracted content is empty")
        
        # Return the extracted content and attachment models
        response =  Response(
            success=True,
            data=PDFResponse(
                metadata=Metadata(
                    url=str(request.url),
                    creation_date=extractor.createDate,
                    modified_date=extractor.modDate,
                    file_name=extractor.filename,
                    file_type=extractor.filetype,
                    file_size=extractor.file_size,
                    num_pages=extractor.num_pages,
                    processing_time=round(extractor.time_taken,3),
                    content_summary=ContentSummary(
                        plain_text_length=len(content_model.plain),
                        html_length=len(content_model.html),
                        markdown_length=len(content_model.markdown),
                        num_attachments=len(attachment_model) if attachment_model else 0
                    )
                ),
                content=content_model,
                attachments=attachment_model,
            )
        )
        
        return JSONResponse(
            status_code=200,
            content=response.model_dump()
        )
        
        
    except urllib.error.URLError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": 400,
                    "message": f"URL Error: {str(e)}"
                }
            }
        )
        
    except HTTPException as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": 400,
                    "message": f"Error processing PDF: {str(e.detail)}"
                }
            }
        )
        

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": f"Error processing PDF: {str(e)}"
                }
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)