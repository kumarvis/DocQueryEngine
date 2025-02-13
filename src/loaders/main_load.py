import requests
import logging
import ftfy
import fs_utils.file_system_utility as fsutils
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "msg",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "hs",
    "lhs",
]


class Loader: 
    def __init__(self):
        pass

    def load(self, file_path: str):
        loader = self._get_loader(file_path)
        docs = loader.load()
        return [
            Document(
                page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
            )
            for doc in docs
        ]

    def _get_loader(self, file_path: str):
        doc_name, ext = fsutils.get_file_name_and_extension(file_path)
        file_ext = ext.lower()
        if file_ext == "pdf":
            loader = PyPDFLoader(
                file_path)
            
        return loader

# class Loader:
#     def __init__(self, engine: str = "", **kwargs):
#         self.engine = engine
#         self.kwargs = kwargs

#     def load(
#         self, filename: str, file_content_type: str, file_path: str
#     ) -> list[Document]:
#         loader = self._get_loader(filename, file_content_type, file_path)
#         docs = loader.load()

#         return [
#             Document(
#                 page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
#             )
#             for doc in docs
#         ]

#     def _get_loader(self, filename: str, file_content_type: str, file_path: str):
#         file_ext = filename.split(".")[-1].lower()
#         if file_ext == "pdf":
#             loader = PyPDFLoader(
#                 file_path, extract_images=self.kwargs.get("PDF_EXTRACT_IMAGES")
#             )
#         elif file_ext == "csv":
#             loader = CSVLoader(file_path)
#         elif file_ext == "rst":
#             loader = UnstructuredRSTLoader(file_path, mode="elements")
#         elif file_ext == "xml":
#             loader = UnstructuredXMLLoader(file_path)
#         elif file_ext in ["htm", "html"]:
#             loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
#         elif file_ext == "md":
#             loader = UnstructuredMarkdownLoader(file_path)
#         elif file_content_type == "application/epub+zip":
#             loader = UnstructuredEPubLoader(file_path)
#         elif (
#             file_content_type
#             == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#             or file_ext == "docx"
#         ):
#             loader = Docx2txtLoader(file_path)
#         elif file_content_type in [
#             "application/vnd.ms-excel",
#             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         ] or file_ext in ["xls", "xlsx"]:
#             loader = UnstructuredExcelLoader(file_path)
#         elif file_content_type in [
#             "application/vnd.ms-powerpoint",
#             "application/vnd.openxmlformats-officedocument.presentationml.presentation",
#         ] or file_ext in ["ppt", "pptx"]:
#             loader = UnstructuredPowerPointLoader(file_path)
#         elif file_ext == "msg":
#             loader = OutlookMessageLoader(file_path)
#         elif file_ext in known_source_ext or (
#             file_content_type and file_content_type.find("text/") >= 0
#         ):
#             loader = TextLoader(file_path, autodetect_encoding=True)
#         else:
#             loader = TextLoader(file_path, autodetect_encoding=True)

#         return loader