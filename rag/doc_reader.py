from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)

def get_documents(doc_dir: str) -> list:
	all_docs = []

	# Читаем текстовые файлы (loader_kwargs передаёт аргументы в TextLoader)
	md_loader = DirectoryLoader(
		doc_dir, glob="**/*.txt",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8"}
	)
	all_docs.extend(md_loader.load())

	# Читаем Markdown файлы (loader_kwargs передаёт аргументы в TextLoader)
	md_loader = DirectoryLoader(
		doc_dir, glob="**/*.md",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8"}
	)
	all_docs.extend(md_loader.load())

	# Читаем PDF файлы (требуется установить PyPDF pip install pypdf)
	pdf_loader = DirectoryLoader(
		doc_dir, glob="**/*.pdf",
		loader_cls=PyPDFLoader
	)
	all_docs.extend(pdf_loader.load())

	return all_docs