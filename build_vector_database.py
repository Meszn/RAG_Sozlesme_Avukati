"""
Vector Database Builder
Faiss library

"""
#import libraries 
import os 
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle 


#program icin dosya olarak .pdf yükleyelim 
# .pdfden metin donusumu yapalim 
def extract_text_from_pdf(pdf_path):
    """
        pdf dosyasindan metin cikartma 
    
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text

# uzun metni daha kucuk parcalara böl
def chunk_text(text, max_lenght=500):
    """
        metni belirtilen karakter uzunluguna göre böl

    """
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_lenght:
            current += " " + line.strip()
        else:
            chunks.append(current.strip())
            current = line.strip()
    if current:
        chunks.append(current.strip())

    return chunks

# text_dumy = extract_text_from_pdf(".\data\Kapsamli_Yazilim_Sozlesmesi.pdf")
# print(chunk_text(text_dumy, max_lenght=500))

# Sentence transformer ile embedding
model = SentenceTransformer("BAAI/bge-m3")

#pdf yolunu belirt 
pdf_file_path = ".\data\Kapsamli_Yazilim_Sozlesmesi.pdf"

#pdften metin çıkaralım 
text = extract_text_from_pdf(pdf_file_path)

#metni chunklara bölelim
chunks = chunk_text(text, max_lenght=500)

#her chunk icin embedding olusturalim
embeddings = model.encode(chunks)

print(f"embeddings shape: {embeddings.shape}") # (n_chunks, embedding_dim)

#faiss index olustur
dimension = embeddings.shape[1] #vektor boyutu
index = faiss.IndexFlatL2(dimension) # L2 norm (Euclidian distance) kullanarak benzerlik arama 
index.add(np.array(embeddings)) # embeddingleri indexe ekle 

#faiss indexi ve chunkları kaydet
faiss.write_index(index, "data/contracy_index.faiss")
with open("data/contracy_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index ve chunklar kaydedildi.")