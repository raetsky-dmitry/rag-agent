from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunks(doc_list: list, chunks_size: int, chunks_overlap: int) -> list:
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunks_size,
		chunk_overlap=chunks_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],  # разделители по приоритету
	)
	return text_splitter.split_documents(doc_list)